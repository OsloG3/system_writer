package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---- Play vs bots ----
//
// The human sits South at a table with three bots served by the bidding-dt
// sidecar (see bidding-dt/src/bidding_dt/play_server.py). Go owns the board
// state and the per-user history; the sidecar owns the model, deal generation,
// legality and double-dummy par scoring. Every finished board scores IMPs
// versus par from the N-S (human team) perspective.

const (
	defaultBotURL   = "http://127.0.0.1:8081"
	humanSeat       = 2  // South
	maxAuctionCalls = 24 // model sequence cap (MAX_CALLS in bidding-dt)
	playBoardTTL    = 6 * time.Hour
	playRecentMax   = 128 // ranking window: last 128 boards per mode

	botTimeout      = 30 * time.Second  // deal / bid / legal
	botScoreTimeout = 90 * time.Second  // score (cold double-dummy solve)
	botPlayTimeout  = 150 * time.Second // a card choice (pool + DD simulation)
)

var (
	botURL    = defaultBotURL
	botModels = map[int]string{} // bot seat -> sidecar model name ("" = default)

	playMu     sync.Mutex // guards playBoards and playStats
	playBoards = map[string]*PlayBoard{}
	playStats  = map[string]PlayStats{}
)

type PlayContract struct {
	Level    int    `json:"level"`
	Denom    string `json:"denom"` // C D H S N
	Declarer int    `json:"declarer"`
	Penalty  int    `json:"penalty"` // 0 none, 1 doubled, 2 redoubled
}

func (c *PlayContract) String() string {
	if c == nil {
		return "passed out"
	}
	decl := "?"
	if c.Declarer >= 0 && c.Declarer < 4 {
		decl = string("NESW"[c.Declarer])
	}
	pen := ""
	if c.Penalty == 1 {
		pen = "X"
	} else if c.Penalty >= 2 {
		pen = "XX"
	}
	return fmt.Sprintf("%d%s%s by %s", c.Level, c.Denom, pen, decl)
}

type PlayResult struct {
	Contract *PlayContract `json:"contract"` // nil = passed out
	Tricks   *int          `json:"tricks"`   // double-dummy tricks for the contract
	Played   *int          `json:"played"`   // tricks the declarer side actually took
	ParNS    int           `json:"parNs"`
	ScoreNS  int           `json:"scoreNs"`
	Imps     float64       `json:"imps"` // human team (N-S) versus par
}

// PlayCard is one card played, in play order. Card is rank+suit, e.g. "AS", "9H".
type PlayCard struct {
	Seat int    `json:"seat"`
	Card string `json:"card"`
}

// PlayState is the card-play phase of a board (nil while bidding or passed out).
type PlayState struct {
	Contract   *PlayContract `json:"contract"`
	Trump      byte          `json:"-"` // denom suit char, or 'N'
	Declarer   int           `json:"declarer"`
	Dummy      int           `json:"dummy"`
	Opener     int           `json:"opener"`
	Plays      []PlayCard    `json:"plays"`
	DeclTricks int           `json:"declTricks"`
	Done       bool          `json:"done"`
}

type PlayBoard struct {
	mu       sync.Mutex // guards the fields below
	ID       string
	User     string
	Mode     string    // "bid" (auction only, scored vs par) or "play" (cards played out)
	Hands    [4]string // N,E,S,W 'S.H.D.C'
	Dealer   int
	Vuln     int
	Calls    []string
	Legal    []string // legal calls for the human at LegalLen calls
	LegalLen int
	Play     *PlayState
	Result   *PlayResult
	Created  time.Time
}

type PlayRecord struct {
	Ts       string     `json:"ts"`
	Mode     string     `json:"mode"`
	Dealer   int        `json:"dealer"`
	Vuln     int        `json:"vuln"`
	Calls    []string   `json:"calls"`
	Contract string     `json:"contract"`
	Tricks   *int       `json:"tricks"`
	Played   *int       `json:"played"`
	ParNS    int        `json:"parNs"`
	ScoreNS  int        `json:"scoreNs"`
	Imps     float64    `json:"imps"`
	Hands    [4]string  `json:"hands,omitempty"` // all four hands, N,E,S,W
	Plays    []PlayCard `json:"plays,omitempty"` // the card-play sequence
}

// PlayStats is the persisted per-user history (data/play_stats.json).
// Recent holds bid-only boards and RecentPlay full-play boards, each newest
// first and capped at playRecentMax (128) -- the ranking window.
type PlayStats struct {
	Boards     int          `json:"boards"`
	ImpsTotal  float64      `json:"impsTotal"`
	Recent     []PlayRecord `json:"recent"`
	RecentPlay []PlayRecord `json:"recentPlay,omitempty"`
}

func initPlay() {
	if u := strings.TrimSpace(os.Getenv("BIDDING_DT_URL")); u != "" {
		botURL = strings.TrimSuffix(u, "/")
	}
	seatOf := map[string]int{"N": 0, "E": 1, "S": 2, "W": 3}
	for _, kv := range strings.Split(os.Getenv("PLAY_BOTS"), ",") {
		seat, model, ok := strings.Cut(strings.TrimSpace(kv), "=")
		if !ok {
			continue
		}
		if s, found := seatOf[strings.ToUpper(seat)]; found && s != humanSeat && model != "" {
			botModels[s] = model
		}
	}
	loadPlayStats()
	log.Printf("Play bots: sidecar %s, seat models %v", botURL, botModels)
}

func playStatsPath() string { return filepath.Join(gameDir(), "play_stats.json") }

func loadPlayStats() {
	data, err := os.ReadFile(playStatsPath())
	if err != nil {
		return
	}
	var m map[string]PlayStats
	if err := json.Unmarshal(data, &m); err != nil {
		log.Printf("Could not parse play_stats.json: %v", err)
		return
	}
	// One-time shape-up: records saved before the per-mode split all sit in
	// Recent; move the full-play ones (mode "play", or a played-trick count)
	// into RecentPlay and stamp the mode on the rest. Idempotent.
	for u, s := range m {
		var bid []PlayRecord
		for _, r := range s.Recent {
			if r.Mode == "play" || (r.Mode == "" && r.Played != nil) {
				r.Mode = "play"
				s.RecentPlay = append(s.RecentPlay, r)
			} else {
				r.Mode = "bid"
				bid = append(bid, r)
			}
		}
		s.Recent = bid
		if len(s.Recent) > playRecentMax {
			s.Recent = s.Recent[:playRecentMax]
		}
		if len(s.RecentPlay) > playRecentMax {
			s.RecentPlay = s.RecentPlay[:playRecentMax]
		}
		m[u] = s
	}
	playMu.Lock()
	playStats = m
	playMu.Unlock()
	log.Printf("Loaded play stats for %d user(s)", len(m))
}

// savePlayStatsLocked writes the history to disk. Callers must hold playMu.
func savePlayStatsLocked() error {
	return writeJSONFile(playStatsPath(), playStats)
}

func getPlayStats(username string) PlayStats {
	playMu.Lock()
	defer playMu.Unlock()
	return playStats[username]
}

// recordPlayResult folds one finished board into the user's history.
// Called with b.mu held (lock order: board -> playMu).
func recordPlayResult(b *PlayBoard) {
	playMu.Lock()
	defer playMu.Unlock()
	s := playStats[b.User]
	s.Boards++
	s.ImpsTotal += b.Result.Imps
	rec := PlayRecord{
		Ts:       time.Now().UTC().Format(time.RFC3339),
		Mode:     b.Mode,
		Dealer:   b.Dealer,
		Vuln:     b.Vuln,
		Calls:    append([]string(nil), b.Calls...),
		Contract: b.Result.Contract.String(),
		Tricks:   b.Result.Tricks,
		Played:   b.Result.Played,
		ParNS:    b.Result.ParNS,
		ScoreNS:  b.Result.ScoreNS,
		Imps:     b.Result.Imps,
		Hands:    b.Hands,
	}
	if b.Play != nil {
		rec.Plays = append([]PlayCard(nil), b.Play.Plays...)
	}
	if b.Mode == "play" {
		s.RecentPlay = append([]PlayRecord{rec}, s.RecentPlay...)
		if len(s.RecentPlay) > playRecentMax {
			s.RecentPlay = s.RecentPlay[:playRecentMax]
		}
	} else {
		s.Recent = append([]PlayRecord{rec}, s.Recent...)
		if len(s.Recent) > playRecentMax {
			s.Recent = s.Recent[:playRecentMax]
		}
	}
	playStats[b.User] = s
	if err := savePlayStatsLocked(); err != nil {
		log.Printf("Could not persist play stats: %v", err)
	}
}

// ---- Sidecar client ----

func botPost(path string, body any, out any, timeout time.Duration) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, botURL+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("unreachable: %v", err)
	}
	defer resp.Body.Close()
	respData, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		msg := ""
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(respData, &e) == nil {
			msg = e.Error
		}
		if msg == "" {
			msg = fmt.Sprintf("sidecar returned %d", resp.StatusCode)
		}
		return errors.New(msg)
	}
	return json.Unmarshal(respData, out)
}

func botUnavailableMsg(err error) string {
	return "bot service error: " + err.Error()
}

// ---- Board logic (callers hold b.mu) ----

func callsOrEmpty(c []string) []string {
	if c == nil {
		return []string{}
	}
	return c
}

func (b *PlayBoard) nextSeat() int { return (b.Dealer + len(b.Calls)) % 4 }

func (b *PlayBoard) over() bool {
	n := len(b.Calls)
	return n >= 4 && b.Calls[n-1] == "P" && b.Calls[n-2] == "P" && b.Calls[n-3] == "P"
}

// advance runs the bots until it is the human's turn or the auction ends,
// then either starts the card play (a contract) or scores a passed-out board.
func (b *PlayBoard) advance() error {
	if b.Play != nil {
		return b.advancePlay()
	}
	for !b.over() && len(b.Calls) < maxAuctionCalls && b.nextSeat() != humanSeat {
		body := map[string]any{
			"hands":  b.Hands,
			"dealer": b.Dealer,
			"vuln":   b.Vuln,
			"calls":  callsOrEmpty(b.Calls),
		}
		if m := botModels[b.nextSeat()]; m != "" {
			body["model"] = m
		}
		var out struct {
			Seat int    `json:"seat"`
			Call string `json:"call"`
		}
		if err := botPost("/bid", body, &out, botTimeout); err != nil {
			return err
		}
		if out.Seat != b.nextSeat() || normalizeCall(out.Call) == "" {
			return errors.New("sidecar returned an inconsistent bid")
		}
		b.Calls = append(b.Calls, out.Call)
	}
	if b.over() {
		if b.Mode == "play" {
			return b.startPlayOrScore()
		}
		return b.score()
	}
	if len(b.Calls) >= maxAuctionCalls {
		return b.score()
	}
	return b.ensureLegal()
}

func (b *PlayBoard) score() error {
	if b.Result != nil {
		return nil
	}
	var out struct {
		Imps     float64       `json:"imps"`
		ParNS    int           `json:"par_ns"`
		ScoreNS  int           `json:"score_ns"`
		Tricks   *int          `json:"tricks"`
		Contract *PlayContract `json:"contract"`
	}
	body := map[string]any{
		"hands":  b.Hands,
		"dealer": b.Dealer,
		"vuln":   b.Vuln,
		"calls":  callsOrEmpty(b.Calls),
	}
	if err := botPost("/score", body, &out, botScoreTimeout); err != nil {
		return err
	}
	b.Result = &PlayResult{
		Contract: out.Contract, Tricks: out.Tricks,
		ParNS: out.ParNS, ScoreNS: out.ScoreNS, Imps: out.Imps,
	}
	recordPlayResult(b)
	return nil
}

// ---- Card play ----
//
// Once the auction ends in a contract the board is played out trick by trick.
// The human keeps South; a bot fills every other seat. The declarer-side bot
// plays both declarer and dummy (it sees both those hands), and each defender
// bot plays itself -- all served by the sidecar's /play/choose, which picks a
// card by double-dummy simulation over hidden hands consistent with the auction
// and the play so far (see bidding-dt/src/bidding_dt/play_bot.py).

const cardRanks = "AKQJT98765432"

// handCards splits a 'S.H.D.C' hand into rank+suit card tokens (e.g. "AS","9S").
func handCards(hand string) []string {
	var out []string
	for s, suit := range strings.Split(hand, ".") {
		if s >= 4 {
			break
		}
		for _, r := range suit {
			out = append(out, string(r)+string("SHDC"[s]))
		}
	}
	return out
}

func cardSuit(card string) byte { return card[1] }

func cardRankIdx(card string) int { return strings.IndexByte(cardRanks, card[0]) }

// normalizeCard maps user/bot input onto a rank+suit token ("AS"); "" if invalid.
func normalizeCard(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	if len(s) == 2 && strings.IndexByte(cardRanks, s[0]) >= 0 && strings.IndexByte("SHDC", s[1]) >= 0 {
		return s
	}
	// tolerate suit+rank ("SA") too
	if len(s) == 2 && strings.IndexByte("SHDC", s[0]) >= 0 && strings.IndexByte(cardRanks, s[1]) >= 0 {
		return string(s[1]) + string(s[0])
	}
	return ""
}

// trickWinner returns the seat that wins a completed trick (4 cards in play
// order). trump is a suit char or 'N'. Mirrors endplay's trick_winner.
func trickWinner(trick []PlayCard, trump byte) int {
	win := trick[0]
	for _, pc := range trick[1:] {
		switch {
		case cardSuit(pc.Card) == cardSuit(win.Card):
			if cardRankIdx(pc.Card) < cardRankIdx(win.Card) {
				win = pc
			}
		case cardSuit(pc.Card) == trump:
			win = pc
		}
	}
	return win.Seat
}

// contractFromCalls recovers the final contract from an auction in seat order,
// mirroring dd/reward.py's contract_from_auction. nil = passed out.
func contractFromCalls(dealer int, calls []string) *PlayContract {
	lastBid := -1
	doubled, redoubled := false, false
	firstDenom := map[[2]int]int{} // [denomIdx, side] -> absolute seat
	for i, raw := range calls {
		c := normalizeCall(raw)
		switch {
		case c == "X":
			doubled = true
		case c == "XX":
			redoubled = true
		case len(c) == 2 && c[0] >= '1' && c[0] <= '7' && strings.IndexByte("CDHSN", c[1]) >= 0:
			lastBid = i
			doubled, redoubled = false, false
			seat := (dealer + i) % 4
			key := [2]int{strings.IndexByte("CDHSN", c[1]), seat % 2}
			if _, ok := firstDenom[key]; !ok {
				firstDenom[key] = seat
			}
		}
	}
	if lastBid < 0 {
		return nil
	}
	c := normalizeCall(calls[lastBid])
	denom := strings.IndexByte("CDHSN", c[1])
	bidder := (dealer + lastBid) % 4
	penalty := 0
	if redoubled {
		penalty = 2
	} else if doubled {
		penalty = 1
	}
	return &PlayContract{
		Level:    int(c[0] - '0'),
		Denom:    string(c[1]),
		Declarer: firstDenom[[2]int{denom, bidder % 2}],
		Penalty:  penalty,
	}
}

// startPlayOrScore begins the card play for a contract, or scores a pass-out.
func (b *PlayBoard) startPlayOrScore() error {
	contract := contractFromCalls(b.Dealer, b.Calls)
	if contract == nil || contract.Declarer < 0 || contract.Declarer > 3 {
		return b.score() // passed out (or truncated): fall back to par scoring
	}
	trump := contract.Denom[0]
	b.Play = &PlayState{
		Contract: contract,
		Trump:    trump,
		Declarer: contract.Declarer,
		Dummy:    (contract.Declarer + 2) % 4,
		Opener:   (contract.Declarer + 1) % 4,
	}
	// Do not advance here: the client paces the play, fetching one bot card at
	// a time (see advancePlay's budget) so each card is revealed as it is
	// computed instead of after every remaining bot has been simulated.
	return nil
}

// nextSeat returns the seat to play next (winner of the last trick leads).
func (p *PlayState) nextSeat() int {
	n := len(p.Plays)
	if n == 0 {
		return p.Opener
	}
	if n%4 == 0 {
		return trickWinner(p.Plays[n-4:], p.Trump)
	}
	return (p.Plays[n-1].Seat + 1) % 4
}

func (p *PlayState) completedTricks() int { return len(p.Plays) / 4 }

// humanControlsPlay reports whether the human (not a bot) plays `seat`.
func (b *PlayBoard) humanControlsPlay(seat int) bool {
	p := b.Play
	if p.Declarer == humanSeat {
		return seat == p.Declarer || seat == p.Dummy
	}
	if p.Dummy == humanSeat {
		return false // the human is dummy; the declarer bot tables those cards
	}
	return seat == humanSeat // the human is a defender
}

// humanPlaySeats lists the seats the human plays this board (for the UI).
func (b *PlayBoard) humanPlaySeats() []int {
	p := b.Play
	if p.Declarer == humanSeat {
		return []int{p.Declarer, p.Dummy}
	}
	if p.Dummy == humanSeat {
		return nil
	}
	return []int{humanSeat}
}

// remainingHand returns the cards `seat` has not yet played.
func (b *PlayBoard) remainingHand(seat int) []string {
	played := map[string]bool{}
	for _, pc := range b.Play.Plays {
		if pc.Seat == seat {
			played[pc.Card] = true
		}
	}
	var out []string
	for _, c := range handCards(b.Hands[seat]) {
		if !played[c] {
			out = append(out, c)
		}
	}
	return out
}

// legalCards returns the cards `seat` may play now (follow suit if able).
func (b *PlayBoard) legalCards(seat int) []string {
	rem := b.remainingHand(seat)
	n := len(b.Play.Plays)
	if n%4 == 0 {
		return rem // on lead: anything
	}
	lead := cardSuit(b.Play.Plays[(n/4)*4].Card)
	var follow []string
	for _, c := range rem {
		if cardSuit(c) == lead {
			follow = append(follow, c)
		}
	}
	if len(follow) > 0 {
		return follow
	}
	return rem
}

// botChoose asks the sidecar for the card a bot seat should play.
func (b *PlayBoard) botChoose(seat int) (string, error) {
	p := b.Play
	var session string
	known := map[string]string{strconv.Itoa(p.Dummy): b.Hands[p.Dummy]}
	if seat == p.Declarer || seat == p.Dummy {
		session = b.ID + ":decl"
		known[strconv.Itoa(p.Declarer)] = b.Hands[p.Declarer]
	} else {
		session = b.ID + ":def" + strconv.Itoa(seat)
		known[strconv.Itoa(seat)] = b.Hands[seat]
	}
	plays := make([][2]any, 0, len(p.Plays))
	for _, pc := range p.Plays {
		plays = append(plays, [2]any{pc.Seat, pc.Card})
	}
	body := map[string]any{
		"session": session,
		"dealer":  b.Dealer,
		"vuln":    b.Vuln,
		"calls":   callsOrEmpty(b.Calls),
		"known":   known,
		"plays":   plays,
		"to_act":  seat,
	}
	var out struct {
		Seat int    `json:"seat"`
		Card string `json:"card"`
	}
	if err := botPost("/play/choose", body, &out, botPlayTimeout); err != nil {
		return "", err
	}
	card := normalizeCard(out.Card)
	if card == "" {
		return "", errors.New("sidecar returned an invalid card")
	}
	return card, nil
}

// applyPlay validates and records one card, updating the trick tally.
func (b *PlayBoard) applyPlay(seat int, card string) error {
	p := b.Play
	if seat != p.nextSeat() {
		return errors.New("played out of turn")
	}
	if !slices.Contains(b.legalCards(seat), card) {
		return fmt.Errorf("illegal card %s for seat %d", card, seat)
	}
	p.Plays = append(p.Plays, PlayCard{Seat: seat, Card: card})
	if len(p.Plays)%4 == 0 {
		w := trickWinner(p.Plays[len(p.Plays)-4:], p.Trump)
		if w == p.Declarer || w == p.Dummy {
			p.DeclTricks++
		}
	}
	if len(p.Plays) == 52 {
		p.Done = true
	}
	return nil
}

// advancePlay runs the bots until the human must play, the hand is over, or a
// single bot play has been made. Stepping one card at a time lets the client
// reveal each card as it is computed (with its own pacing) rather than waiting
// for a whole trick of expensive double-dummy simulations to finish. When the
// human is the dummy every seat is a bot, so the client keeps polling to watch
// the hand unfold one card per request.
func (b *PlayBoard) advancePlay() error {
	p := b.Play
	if p == nil || b.Result != nil {
		return nil
	}
	budget := 1
	for !p.Done && budget > 0 {
		seat := p.nextSeat()
		if b.humanControlsPlay(seat) {
			return nil // wait for the human
		}
		card, err := b.botChoose(seat)
		if err != nil {
			return err
		}
		if err := b.applyPlay(seat, card); err != nil {
			return err
		}
		budget--
	}
	if p.Done {
		return b.finishPlay()
	}
	return nil
}

// finishPlay scores the actually-played tricks against double-dummy par.
func (b *PlayBoard) finishPlay() error {
	if b.Result != nil {
		return nil
	}
	var out struct {
		Contract *PlayContract `json:"contract"`
		Tricks   *int          `json:"tricks"`
		DDTricks *int          `json:"dd_tricks"`
		ParNS    int           `json:"par_ns"`
		ScoreNS  int           `json:"score_ns"`
		Imps     float64       `json:"imps"`
	}
	body := map[string]any{
		"hands":  b.Hands,
		"dealer": b.Dealer,
		"vuln":   b.Vuln,
		"calls":  callsOrEmpty(b.Calls),
		"tricks": b.Play.DeclTricks,
	}
	if err := botPost("/play/result", body, &out, botScoreTimeout); err != nil {
		return err
	}
	played := b.Play.DeclTricks
	b.Result = &PlayResult{
		Contract: out.Contract, Tricks: out.DDTricks, Played: &played,
		ParNS: out.ParNS, ScoreNS: out.ScoreNS, Imps: out.Imps,
	}
	recordPlayResult(b)
	b.dropPlaySessions()
	return nil
}

// dropPlaySessions tells the sidecar to forget this board's card-play sessions.
func (b *PlayBoard) dropPlaySessions() {
	var out struct{}
	_ = botPost("/play/drop", map[string]any{"prefix": b.ID + ":"}, &out, botTimeout)
}

func (b *PlayBoard) ensureLegal() error {
	if b.Result != nil {
		b.Legal = nil
		b.LegalLen = -1
		return nil
	}
	if b.Legal != nil && b.LegalLen == len(b.Calls) {
		return nil
	}
	var out struct {
		Seat  int      `json:"seat"`
		Over  bool     `json:"over"`
		Legal []string `json:"legal"`
	}
	body := map[string]any{"dealer": b.Dealer, "vuln": b.Vuln, "calls": callsOrEmpty(b.Calls)}
	if err := botPost("/legal", body, &out, botTimeout); err != nil {
		return err
	}
	b.Legal = out.Legal
	b.LegalLen = len(b.Calls)
	return nil
}

// stateLocked renders the client-visible board. During the auction only the
// human's hand is exposed; during the play the human's hand and the dummy are
// face up; all four are revealed once the board is scored.
func (b *PlayBoard) stateLocked(stats PlayStats) map[string]any {
	st := map[string]any{
		"id":        b.ID,
		"dealer":    b.Dealer,
		"vuln":      b.Vuln,
		"humanSeat": humanSeat,
		"boardMode": b.Mode,
		"hand":      b.Hands[humanSeat],
		"calls":     callsOrEmpty(b.Calls),
		"bots":      botLabels(),
		"stats":     statsSummary(stats),
		"done":      b.Result != nil,
	}
	switch {
	case b.Result != nil:
		st["phase"] = "done"
		st["result"] = b.Result
		st["hands"] = b.Hands
		st["yourTurn"] = false
		st["nextSeat"] = -1
		if b.Play != nil {
			st["play"] = b.playStateLocked(false)
		}
	case b.Play != nil:
		st["phase"] = "play"
		seat := b.Play.nextSeat()
		st["nextSeat"] = seat
		st["yourTurn"] = b.humanControlsPlay(seat)
		st["play"] = b.playStateLocked(true)
	default:
		st["phase"] = "bid"
		st["nextSeat"] = b.nextSeat()
		st["yourTurn"] = b.nextSeat() == humanSeat
		if b.nextSeat() == humanSeat && b.Legal != nil && b.LegalLen == len(b.Calls) {
			st["legal"] = b.Legal
		}
	}
	// Alerts from the model's self-play tree: what each bot bid promised and
	// what each of the human's legal calls would show. Both are omitted
	// until the tree finishes loading in the background.
	if t := alertTreeReady(); t != nil {
		if a := t.callAlerts(b.Dealer, b.Calls, 1<<humanSeat); a != nil {
			st["alerts"] = a
		}
		if st["legal"] != nil {
			if a := t.optionAlerts(b.Calls, b.Legal); a != nil {
				st["optionAlerts"] = a
			}
		}
	}
	return st
}

// playStateLocked renders the card-play phase for the client. The human's own
// hand travels in stateLocked("hand"); the dummy is face up here. Hidden seats
// are exposed only as a card count.
func (b *PlayBoard) playStateLocked(withLegal bool) map[string]any {
	p := b.Play
	seat := p.nextSeat()
	plays := p.Plays
	if plays == nil {
		plays = []PlayCard{}
	}
	humanSeats := b.humanPlaySeats()
	if humanSeats == nil {
		humanSeats = []int{}
	}
	counts := [4]int{}
	for s := 0; s < 4; s++ {
		counts[s] = len(b.remainingHand(s))
	}
	m := map[string]any{
		"contract":   p.Contract,
		"declarer":   p.Declarer,
		"dummy":      p.Dummy,
		"opener":     p.Opener,
		"trump":      string(p.Trump),
		"plays":      plays,
		"nextSeat":   seat,
		"declTricks": p.DeclTricks,
		"defTricks":  p.completedTricks() - p.DeclTricks,
		"humanSeats": humanSeats,
		"dummyHand":  b.Hands[p.Dummy],
		"counts":     counts,
		"done":       p.Done,
		"yourTurn":   false,
	}
	if withLegal && b.humanControlsPlay(seat) {
		m["yourTurn"] = true
		m["actSeat"] = seat
		m["legal"] = b.legalCards(seat)
	}
	return m
}

func botLabels() map[string]string {
	m := make(map[string]string, len(botModels))
	for seat, name := range botModels {
		m[strconv.Itoa(seat)] = name
	}
	return m
}

func statsSummary(s PlayStats) map[string]any {
	avg := 0.0
	if s.Boards > 0 {
		avg = s.ImpsTotal / float64(s.Boards)
	}
	recent := s.Recent
	if recent == nil {
		recent = []PlayRecord{}
	}
	recentPlay := s.RecentPlay
	if recentPlay == nil {
		recentPlay = []PlayRecord{}
	}
	return map[string]any{
		"boards":     s.Boards,
		"impsTotal":  s.ImpsTotal,
		"avgImps":    avg,
		"recent":     recent,
		"recentPlay": recentPlay,
	}
}

// normalizeCall maps user/bot input onto the canonical vocab tokens
// (P, X, XX, 1C..7N); returns "" when the input is not a call.
func normalizeCall(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	switch s {
	case "PASS":
		s = "P"
	case "DBL", "DOUBLE":
		s = "X"
	case "RDBL", "REDOUBLE":
		s = "XX"
	}
	if s == "P" || s == "X" || s == "XX" {
		return s
	}
	if len(s) == 3 && s[1:] == "NT" {
		s = s[:1] + "N"
	}
	if len(s) == 2 && s[0] >= '1' && s[0] <= '7' && strings.IndexByte("CDHSN", s[1]) >= 0 {
		return s
	}
	return ""
}

func lookupPlayBoard(id, username string) *PlayBoard {
	playMu.Lock()
	defer playMu.Unlock()
	b := playBoards[id]
	if b == nil || b.User != username {
		return nil
	}
	return b
}

func gcPlayBoardsLocked() {
	now := time.Now()
	for id, b := range playBoards {
		if now.Sub(b.Created) > playBoardTTL {
			delete(playBoards, id)
		}
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// ---- Handlers ----

// POST /api/play/new - deals a fresh board and runs the bots up to the human.
// Body (optional): {"mode":"bid"|"play"} -- "bid" scores the auction against
// double-dummy par (no cards), "play" plays the whole hand out. Default "bid".
func handlePlayNew(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	mode := "bid"
	var body struct {
		Mode string `json:"mode"`
	}
	limitBody(w, r, maxBodyAuth)
	if err := json.NewDecoder(r.Body).Decode(&body); err == nil && body.Mode == "play" {
		mode = "play"
	}
	var deal struct {
		Hands  [4]string `json:"hands"`
		Dealer int       `json:"dealer"`
		Vuln   int       `json:"vuln"`
	}
	if err := botPost("/deal", struct{}{}, &deal, botTimeout); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	b := &PlayBoard{
		ID: generateID(), User: username, Mode: mode, Hands: deal.Hands,
		Dealer: deal.Dealer, Vuln: deal.Vuln,
		LegalLen: -1, Created: time.Now(),
	}
	if err := b.advance(); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	playMu.Lock()
	gcPlayBoardsLocked()
	playBoards[b.ID] = b
	stats := playStats[username]
	playMu.Unlock()

	b.mu.Lock()
	st := b.stateLocked(stats)
	b.mu.Unlock()
	writeJSON(w, st)
}

// GET /api/play/{id} - current board state (re-advances stalled bot turns)
func handlePlayGet(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid board ID")
		return
	}
	b := lookupPlayBoard(id, username)
	if b == nil {
		jsonError(w, http.StatusNotFound, "no such board")
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Result == nil {
		if err := b.advance(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
	}
	writeJSON(w, b.stateLocked(getPlayStats(username)))
}

// POST /api/play/{id}/call - the human's call, then bots until the human is
// up again or the auction ends (scored and recorded once)
func handlePlayCall(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid board ID")
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Call string `json:"call"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	call := normalizeCall(body.Call)
	if call == "" {
		jsonError(w, http.StatusBadRequest, "invalid call")
		return
	}
	b := lookupPlayBoard(id, username)
	if b == nil {
		jsonError(w, http.StatusNotFound, "no such board")
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Result != nil {
		jsonError(w, http.StatusConflict, "board is finished")
		return
	}
	if b.Play != nil {
		jsonError(w, http.StatusConflict, "the auction is over; play a card")
		return
	}
	if b.nextSeat() != humanSeat {
		// recover a turn that stalled on a sidecar failure
		if err := b.advance(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
		if b.Result != nil {
			writeJSON(w, b.stateLocked(getPlayStats(username)))
			return
		}
	}
	if err := b.ensureLegal(); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	if !slices.Contains(b.Legal, call) {
		jsonError(w, http.StatusBadRequest, "illegal call: "+call)
		return
	}
	b.Calls = append(b.Calls, call)
	b.Legal = nil
	b.LegalLen = -1
	if err := b.advance(); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	writeJSON(w, b.stateLocked(getPlayStats(username)))
}

// POST /api/play/{id}/card - the human plays a card. The response returns
// immediately with that card recorded; the bots then advance one card per
// subsequent GET (see advancePlay) so the client can reveal them one at a time.
func handlePlayCard(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid board ID")
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Card string `json:"card"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	card := normalizeCard(body.Card)
	if card == "" {
		jsonError(w, http.StatusBadRequest, "invalid card")
		return
	}
	b := lookupPlayBoard(id, username)
	if b == nil {
		jsonError(w, http.StatusNotFound, "no such board")
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Result != nil {
		jsonError(w, http.StatusConflict, "board is finished")
		return
	}
	if b.Play == nil {
		// the auction may not have reached the play yet (a stalled bot turn)
		if err := b.advance(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
		if b.Play == nil {
			jsonError(w, http.StatusConflict, "the auction is not over")
			return
		}
	}
	seat := b.Play.nextSeat()
	if !b.humanControlsPlay(seat) {
		// recover a bot turn that stalled on a sidecar failure
		if err := b.advancePlay(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
		writeJSON(w, b.stateLocked(getPlayStats(username)))
		return
	}
	if !slices.Contains(b.legalCards(seat), card) {
		jsonError(w, http.StatusBadRequest, "illegal card: "+card)
		return
	}
	if err := b.applyPlay(seat, card); err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}
	// Return straight away so the human sees their own card land immediately;
	// the client then paces the bots one card per request (see advancePlay).
	writeJSON(w, b.stateLocked(getPlayStats(username)))
}

// GET /api/play/stats - the user's history vs the bots
func handlePlayStats(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	writeJSON(w, statsSummary(getPlayStats(username)))
}
