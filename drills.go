package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
)

// ---- Card problems: authored card-play drills ----
//
// A problem is a fully scripted board -- deal, auction, comments and every
// card of the play, each hero card carrying the author's "why" message --
// stored per problem in data/drills/<id>.json. Nothing here touches the
// bidding-dt sidecar: legality and scoring are all local.
//
// Practice replays the script: the hero (the declarer side for declarer
// problems, one chosen defender for defense problems) must find the authored
// card at every turn; a different pick is refused and answered with the
// message the author attached to the correct card. Every other seat plays
// itself from the script. The hand ends after 52 cards or at the author's
// claim (a trick count plus a plain final score -- no IMPs, no par).

const (
	drillMaxCalls   = 40 // authored auctions are not bound by the model's cap
	drillSessionTTL = 6 * time.Hour
)

var (
	drillsMu sync.Mutex
	drills   = map[string]*DrillProblem{} // keyed by problem ID

	drillSessMu sync.Mutex
	drillSess   = map[string]*DrillSession{}
)

// DrillPlay is one scripted card. Msg is the author's explanation and is only
// allowed on the seats the hero controls.
type DrillPlay struct {
	Seat int    `json:"seat"`
	Card string `json:"card"`
	Msg  string `json:"msg,omitempty"`
}

// DrillClaim ends the hand early: the declarer's final trick count. The result
// is shown as the contract and its margin (e.g. "4S-1") -- no score, no IMPs.
// nil = all 52 cards are played out.
type DrillClaim struct {
	Tricks int `json:"tricks"`
}

type DrillProblem struct {
	ID             string      `json:"id"`
	Owner          string      `json:"owner"`
	Title          string      `json:"title,omitempty"`
	Role           string      `json:"role"`     // "declarer" | "defense"
	HeroSeat       int         `json:"heroSeat"` // declarer, or the one chosen defender
	Difficulty     int         `json:"difficulty"`
	Dealer         int         `json:"dealer"`
	Vuln           int         `json:"vuln"`
	Hands          [4]string   `json:"hands"` // N,E,S,W 'S.H.D.C'
	Calls          []string    `json:"calls"`
	BidComment     string      `json:"bidComment,omitempty"`
	ProblemComment string      `json:"problemComment,omitempty"`
	Plays          []DrillPlay `json:"plays"`
	Claim          *DrillClaim `json:"claim,omitempty"`
	Created        string      `json:"created"`
	Updated        string      `json:"updated"`
}

type DrillResult struct {
	Contract *PlayContract `json:"contract"`
	Tricks   int           `json:"tricks"`
	Claimed  bool          `json:"claimed"`
	Wrong    int           `json:"wrong"`
}

// DrillSession is one practice run. The problem is snapshotted at the start
// so edits cannot desync a session in flight. In-memory only, like the solo
// play boards.
type DrillSession struct {
	mu         sync.Mutex
	ID         string
	User       string
	Problem    DrillProblem
	contract   *PlayContract
	trump      byte
	Plays      []DrillPlay // applied prefix of the script
	DeclTricks int
	Wrong      int
	Done       bool
	Result     *DrillResult
	Created    time.Time
}

func drillsDir() string          { return filepath.Join(dataDir, "drills") }
func drillPath(id string) string { return filepath.Join(drillsDir(), id+".json") }

func initDrills() {
	if err := os.MkdirAll(drillsDir(), 0o755); err != nil {
		log.Printf("Could not create drills directory: %v", err)
		return
	}
	files, err := filepath.Glob(filepath.Join(drillsDir(), "*.json"))
	if err != nil {
		return
	}
	n := 0
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var p DrillProblem
		if err := json.Unmarshal(data, &p); err != nil {
			log.Printf("Could not parse %s: %v", f, err)
			continue
		}
		if p.ID == "" {
			p.ID = strings.TrimSuffix(filepath.Base(f), ".json")
		}
		if !validID(p.ID) {
			continue
		}
		drills[p.ID] = &p
		n++
	}
	if n > 0 {
		log.Printf("Loaded %d card problem(s)", n)
	}
}

// saveDrillLocked persists a problem. Callers must hold drillsMu.
func saveDrillLocked(p *DrillProblem) error {
	return writeJSONFile(drillPath(p.ID), p)
}

func lookupDrill(id string) *DrillProblem {
	drillsMu.Lock()
	defer drillsMu.Unlock()
	return drills[id]
}

// ---- Validation ----

func validSeat(s int) bool { return s >= 0 && s <= 3 }

// validateHands checks the four 'S.H.D.C' strings form a full 52-card deck
// with 13 cards per seat.
func validateHands(hands [4]string) error {
	seen := map[string]bool{}
	for seat, h := range hands {
		parts := strings.Split(h, ".")
		if len(parts) != 4 {
			return fmt.Errorf("hand %s: want four dot-separated suits (S.H.D.C)", "NESW"[seat:seat+1])
		}
		n := 0
		for s, suit := range parts {
			for _, r := range suit {
				card := string(r) + string("SHDC"[s])
				if normalizeCard(card) != card {
					return fmt.Errorf("hand %s: bad card %q", "NESW"[seat:seat+1], card)
				}
				if seen[card] {
					return fmt.Errorf("card %s appears twice", card)
				}
				seen[card] = true
				n++
			}
		}
		if n != 13 {
			return fmt.Errorf("hand %s has %d cards, want 13", "NESW"[seat:seat+1], n)
		}
	}
	if len(seen) != 52 {
		return errors.New("the four hands must cover all 52 cards")
	}
	return nil
}

func bidRank(c string) int {
	if len(c) == 2 && c[0] >= '1' && c[0] <= '7' && strings.IndexByte("CDHSN", c[1]) >= 0 {
		return int(c[0]-'0')*10 + strings.IndexByte("CDHSN", c[1])
	}
	return -1
}

// validateAuction checks an authored auction is legal and ends in a contract
// (three passes over a bid; passed-out problems are not card-play problems).
// Returns the resulting contract.
func validateAuction(dealer int, calls []string) (*PlayContract, error) {
	if len(calls) < 4 {
		return nil, errors.New("the auction is too short")
	}
	if len(calls) > drillMaxCalls {
		return nil, fmt.Errorf("the auction has more than %d calls", drillMaxCalls)
	}
	lastBidRank, lastBidSeat := -1, -1
	for i, raw := range calls {
		c := normalizeCall(raw)
		if c == "" {
			return nil, fmt.Errorf("call %d is not a valid call", i+1)
		}
		if auctionFinished(calls[:i]) {
			return nil, errors.New("the auction continues after three passes")
		}
		seat := (dealer + i) % 4
		prev := ""
		if i > 0 {
			prev = normalizeCall(calls[i-1])
		}
		switch {
		case c == "P":
		case c == "X":
			// a double must sit directly over an opponent's bid
			if bidRank(prev) < 0 || lastBidSeat < 0 || lastBidSeat%2 == seat%2 {
				return nil, fmt.Errorf("call %d: double must directly follow an opponent's bid", i+1)
			}
		case c == "XX":
			// a redouble must sit directly over the opponents' double of
			// your side's bid
			if prev != "X" || lastBidSeat < 0 || lastBidSeat%2 != seat%2 {
				return nil, fmt.Errorf("call %d: redouble must directly follow the opponents' double of your bid", i+1)
			}
		default: // a bid
			if bidRank(c) <= lastBidRank {
				return nil, fmt.Errorf("call %d: %s does not outrank the previous bid", i+1, c)
			}
			lastBidRank, lastBidSeat = bidRank(c), seat
		}
	}
	if !auctionFinished(calls) {
		return nil, errors.New("the auction must end with a contract and three passes")
	}
	contract := contractFromCalls(dealer, calls)
	if contract == nil {
		return nil, errors.New("the auction must end in a contract, not a pass-out")
	}
	return contract, nil
}

// heroSeatsLocked lists the seats the hero plays for a problem.
func (p *DrillProblem) heroSeats() []int {
	if p.Role == "declarer" {
		return []int{p.HeroSeat, (p.HeroSeat + 2) % 4}
	}
	return []int{p.HeroSeat}
}

// simulateDrillPlays walks the scripted play, enforcing turn order, card
// ownership and follow-suit, and that only hero seats carry messages. It
// returns the declarer side's tricks won by the end of the line.
func simulateDrillPlays(p *DrillProblem, contract *PlayContract) (int, error) {
	dummy := (contract.Declarer + 2) % 4
	hero := p.heroSeats()
	remaining := map[int]map[string]bool{}
	for seat := 0; seat < 4; seat++ {
		remaining[seat] = map[string]bool{}
		for _, c := range handCards(p.Hands[seat]) {
			remaining[seat][c] = true
		}
	}
	declTricks := 0
	next := (contract.Declarer + 1) % 4
	for i, dp := range p.Plays {
		card := normalizeCard(dp.Card)
		if card == "" {
			return 0, fmt.Errorf("play %d: bad card %q", i+1, dp.Card)
		}
		if !validSeat(dp.Seat) {
			return 0, fmt.Errorf("play %d: bad seat", i+1)
		}
		if dp.Seat != next {
			return 0, fmt.Errorf("play %d: %s plays out of turn (expected %s)",
				i+1, "NESW"[dp.Seat:dp.Seat+1], "NESW"[next:next+1])
		}
		isHero := slices.Contains(hero, dp.Seat)
		if strings.TrimSpace(dp.Msg) != "" && !isHero {
			return 0, fmt.Errorf("play %d: only the hero's cards may carry a message", i+1)
		}
		if !remaining[dp.Seat][card] {
			return 0, fmt.Errorf("play %d: %s does not hold %s", i+1, "NESW"[dp.Seat:dp.Seat+1], card)
		}
		if i%4 > 0 {
			lead := cardSuit(p.Plays[(i/4)*4].Card)
			if cardSuit(card) != lead {
				hasLead := false
				for c := range remaining[dp.Seat] {
					if cardSuit(c) == lead {
						hasLead = true
						break
					}
				}
				if hasLead {
					return 0, fmt.Errorf("play %d: %s must follow suit", i+1, "NESW"[dp.Seat:dp.Seat+1])
				}
			}
		}
		delete(remaining[dp.Seat], card)
		p.Plays[i].Card = card
		if (i+1)%4 == 0 {
			trick := make([]PlayCard, 4)
			for j := 0; j < 4; j++ {
				trick[j] = PlayCard{Seat: p.Plays[i-3+j].Seat, Card: p.Plays[i-3+j].Card}
			}
			if w := trickWinner(trick, contract.Denom[0]); w == contract.Declarer || w == dummy {
				declTricks++
			}
			next = trickWinner(trick, contract.Denom[0])
		} else {
			next = (dp.Seat + 1) % 4
		}
	}
	return declTricks, nil
}

// validateDrill checks a problem as authored: the deal, the auction, the hero
// setup and the whole scripted play, including the claim bounds.
func validateDrill(p *DrillProblem) (*PlayContract, error) {
	if p.Role != "declarer" && p.Role != "defense" {
		return nil, errors.New(`role must be "declarer" or "defense"`)
	}
	if p.Difficulty < 1 || p.Difficulty > 5 {
		return nil, errors.New("difficulty must be between 1 and 5 stars")
	}
	if !validSeat(p.Dealer) || !validSeat(p.Vuln) || !validSeat(p.HeroSeat) {
		return nil, errors.New("dealer, vulnerability and hero seat must be valid seats")
	}
	if err := validateHands(p.Hands); err != nil {
		return nil, err
	}
	for i, c := range p.Calls {
		n := normalizeCall(c)
		if n == "" {
			return nil, fmt.Errorf("call %d is not a valid call", i+1)
		}
		p.Calls[i] = n
	}
	contract, err := validateAuction(p.Dealer, p.Calls)
	if err != nil {
		return nil, err
	}
	defenders := []int{(contract.Declarer + 1) % 4, (contract.Declarer + 3) % 4}
	if p.Role == "declarer" {
		if p.HeroSeat != contract.Declarer {
			return nil, errors.New("a declarer problem's hero must be the declarer")
		}
	} else if !slices.Contains(defenders, p.HeroSeat) {
		return nil, errors.New("a defense problem's hero must be one of the defenders")
	}
	if len(p.Plays) == 0 {
		return nil, errors.New("record the card play first")
	}
	declTricks, err := simulateDrillPlays(p, contract)
	if err != nil {
		return nil, err
	}
	if p.Claim == nil {
		if len(p.Plays) != 52 {
			return nil, errors.New("the play must cover all 52 cards, or end with a claim")
		}
	} else {
		if len(p.Plays)%4 != 0 || len(p.Plays) < 4 {
			return nil, errors.New("a claim must come at the end of a complete trick")
		}
		remaining := (52 - len(p.Plays)) / 4
		if p.Claim.Tricks < declTricks || p.Claim.Tricks > declTricks+remaining {
			return nil, fmt.Errorf("the claimed %d tricks are impossible (%d already won, %d still to play)",
				p.Claim.Tricks, declTricks, remaining)
		}
	}
	p.Title = strings.TrimSpace(p.Title)
	p.BidComment = strings.TrimSpace(p.BidComment)
	p.ProblemComment = strings.TrimSpace(p.ProblemComment)
	return contract, nil
}

var cardRankNames = map[byte]string{
	'A': "A", 'K': "K", 'Q': "Q", 'J': "J", 'T': "10",
	'9': "9", '8': "8", '7': "7", '6': "6", '5': "5",
	'4': "4", '3': "3", '2': "2",
}

// drillCardName renders a card token for prose, e.g. "AS" -> "A\u2660".
func drillCardName(card string) string {
	if len(card) != 2 {
		return card
	}
	suit := map[byte]string{'S': "\u2660", 'H': "\u2665", 'D': "\u2666", 'C': "\u2663"}[card[1]]
	return cardRankNames[card[0]] + suit
}

// defaultHeroMsg is the message the client pre-fills (and the server falls
// back to) for a hero card the author did not annotate.
func defaultHeroMsg(card string) string {
	return fmt.Sprintf("I just decided to play the %s instead.", drillCardName(card))
}

// ---- Session logic (callers hold s.mu) ----

func (s *DrillSession) heroControls(seat int) bool {
	return slices.Contains(s.Problem.heroSeats(), seat)
}

func (s *DrillSession) nextSeatLocked() int {
	n := len(s.Plays)
	if n == 0 {
		return (s.contract.Declarer + 1) % 4
	}
	if n%4 == 0 {
		trick := make([]PlayCard, 4)
		for j := 0; j < 4; j++ {
			trick[j] = PlayCard{Seat: s.Plays[n-4+j].Seat, Card: s.Plays[n-4+j].Card}
		}
		return trickWinner(trick, s.trump)
	}
	return (s.Plays[n-1].Seat + 1) % 4
}

func (s *DrillSession) remainingHandLocked(seat int) []string {
	played := map[string]bool{}
	for _, dp := range s.Plays {
		if dp.Seat == seat {
			played[dp.Card] = true
		}
	}
	var out []string
	for _, c := range handCards(s.Problem.Hands[seat]) {
		if !played[c] {
			out = append(out, c)
		}
	}
	return out
}

func (s *DrillSession) legalCardsLocked(seat int) []string {
	rem := s.remainingHandLocked(seat)
	n := len(s.Plays)
	if n%4 == 0 {
		return rem
	}
	lead := cardSuit(s.Plays[(n/4)*4].Card)
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

func (s *DrillSession) applyLocked(dp DrillPlay) {
	s.Plays = append(s.Plays, dp)
	if len(s.Plays)%4 == 0 {
		n := len(s.Plays)
		trick := make([]PlayCard, 4)
		for j := 0; j < 4; j++ {
			trick[j] = PlayCard{Seat: s.Plays[n-4+j].Seat, Card: s.Plays[n-4+j].Card}
		}
		w := trickWinner(trick, s.trump)
		if w == s.contract.Declarer || w == (s.contract.Declarer+2)%4 {
			s.DeclTricks++
		}
	}
	if len(s.Plays) == len(s.Problem.Plays) {
		s.finishLocked()
	}
}

func (s *DrillSession) finishLocked() {
	if s.Done {
		return
	}
	s.Done = true
	if c := s.Problem.Claim; c != nil {
		s.Result = &DrillResult{
			Contract: s.contract, Tricks: c.Tricks,
			Claimed: true, Wrong: s.Wrong,
		}
		return
	}
	s.Result = &DrillResult{
		Contract: s.contract, Tricks: s.DeclTricks,
		Wrong: s.Wrong,
	}
}

// advanceLocked plays every scripted card until it is the hero's turn again
// or the hand ends. No simulation anywhere: the author's line is the truth.
func (s *DrillSession) advanceLocked() {
	for !s.Done && len(s.Plays) < len(s.Problem.Plays) {
		seat := s.nextSeatLocked()
		if s.heroControls(seat) {
			return
		}
		s.applyLocked(s.Problem.Plays[len(s.Plays)])
	}
	if len(s.Plays) == len(s.Problem.Plays) {
		s.finishLocked()
	}
}

// stateLocked renders the client-visible session. During the play the hero
// sees their own hand only (plus the dummy once tabled); the script's
// messages stay hidden until the hand is over, except for the hint returned
// with a wrong pick.
func (s *DrillSession) stateLocked() map[string]any {
	p := &s.Problem
	st := map[string]any{
		"mode":           "drill",
		"id":             s.ID,
		"problemId":      p.ID,
		"title":          p.Title,
		"owner":          p.Owner,
		"role":           p.Role,
		"heroSeat":       p.HeroSeat,
		"difficulty":     p.Difficulty,
		"dealer":         p.Dealer,
		"vuln":           p.Vuln,
		"calls":          callsOrEmpty(p.Calls),
		"bidComment":     p.BidComment,
		"problemComment": p.ProblemComment,
		"contract":       s.contract,
		"declarer":       s.contract.Declarer,
		"dummy":          (s.contract.Declarer + 2) % 4,
		"opener":         (s.contract.Declarer + 1) % 4,
		"trump":          string(s.trump),
		"hand":           p.Hands[p.HeroSeat],
		"heroSeats":      p.heroSeats(),
		"claimed":        p.Claim != nil,
		"wrong":          s.Wrong,
		"done":           s.Done,
	}
	plays := make([]map[string]any, 0, len(s.Plays))
	for _, dp := range s.Plays {
		plays = append(plays, map[string]any{"seat": dp.Seat, "card": dp.Card})
	}
	counts := [4]int{}
	for seat := 0; seat < 4; seat++ {
		counts[seat] = len(s.remainingHandLocked(seat))
	}
	dummyHand := ""
	if len(s.Plays) > 0 {
		dummyHand = p.Hands[(s.contract.Declarer+2)%4]
	}
	declTricks := s.DeclTricks
	done := len(s.Plays) / 4
	st["plays"] = plays
	st["declTricks"] = declTricks
	st["defTricks"] = done - declTricks
	st["counts"] = counts
	st["dummyHand"] = dummyHand
	if s.Done {
		st["phase"] = "done"
		st["yourTurn"] = false
		st["nextSeat"] = -1
		st["hands"] = p.Hands
		st["result"] = s.Result
		line := make([]map[string]any, 0, len(p.Plays))
		for _, dp := range p.Plays {
			m := map[string]any{"seat": dp.Seat, "card": dp.Card}
			if s.heroControls(dp.Seat) {
				msg := strings.TrimSpace(dp.Msg)
				if msg == "" {
					msg = defaultHeroMsg(dp.Card)
				}
				m["msg"] = msg
			}
			line = append(line, m)
		}
		st["line"] = line
		return st
	}
	st["phase"] = "play"
	seat := s.nextSeatLocked()
	st["nextSeat"] = seat
	st["yourTurn"] = s.heroControls(seat)
	if st["yourTurn"].(bool) {
		st["actSeat"] = seat
		st["legal"] = s.legalCardsLocked(seat)
	}
	return st
}

func lookupDrillSession(id, username string) *DrillSession {
	drillSessMu.Lock()
	defer drillSessMu.Unlock()
	s := drillSess[id]
	if s == nil || s.User != username {
		return nil
	}
	return s
}

func gcDrillSessionsLocked() {
	now := time.Now()
	for id, s := range drillSess {
		if now.Sub(s.Created) > drillSessionTTL {
			delete(drillSess, id)
		}
	}
}

// newDrillSession snapshots a problem and runs the script up to the hero's
// first turn.
func newDrillSession(p *DrillProblem, username string) (*DrillSession, error) {
	contract, err := validateDrill(p)
	if err != nil {
		return nil, err
	}
	s := &DrillSession{
		ID: generateID(), User: username, Problem: *p,
		contract: contract, trump: contract.Denom[0], Created: time.Now(),
	}
	s.advanceLocked()
	return s, nil
}

// ---- Handlers ----

// GET /api/drills - the library: metadata for every problem, no hands, no
// script, no comments (nothing that could spoil a practice run).
func handleDrillsList(w http.ResponseWriter, r *http.Request) {
	if requireLogin(w, r) == "" {
		return
	}
	drillsMu.Lock()
	type info struct {
		ID         string `json:"id"`
		Owner      string `json:"owner"`
		Title      string `json:"title"`
		Role       string `json:"role"`
		HeroSeat   int    `json:"heroSeat"`
		Difficulty int    `json:"difficulty"`
		Dealer     int    `json:"dealer"`
		Vuln       int    `json:"vuln"`
		Contract   string `json:"contract"`
		Claimed    bool   `json:"claimed"`
		Created    string `json:"created"`
		Updated    string `json:"updated"`
	}
	list := make([]info, 0, len(drills))
	for _, p := range drills {
		label := "passed out"
		if c := contractFromCalls(p.Dealer, p.Calls); c != nil {
			label = c.String()
		}
		list = append(list, info{
			ID: p.ID, Owner: p.Owner, Title: p.Title, Role: p.Role,
			HeroSeat: p.HeroSeat, Difficulty: p.Difficulty, Dealer: p.Dealer,
			Vuln: p.Vuln, Contract: label, Claimed: p.Claim != nil,
			Created: p.Created, Updated: p.Updated,
		})
	}
	drillsMu.Unlock()
	sort.Slice(list, func(i, j int) bool { return list[i].Updated > list[j].Updated })
	writeJSON(w, map[string]any{"problems": list})
}

// GET /api/drills/{id} - the full problem, for its owner's editor only.
func handleDrillGet(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	p := lookupDrill(id)
	if p == nil {
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	if p.Owner != username {
		jsonError(w, http.StatusForbidden, "only the author can open this problem")
		return
	}
	writeJSON(w, p)
}

// decodeDrillBody reads and validates a create/edit payload.
func decodeDrillBody(w http.ResponseWriter, r *http.Request) (*DrillProblem, *PlayContract, bool) {
	limitBody(w, r, maxBodyAuth)
	var p DrillProblem
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return nil, nil, false
	}
	contract, err := validateDrill(&p)
	if err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return nil, nil, false
	}
	return &p, contract, true
}

// POST /api/drills - create a problem owned by the current user.
func handleDrillCreate(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	p, _, ok := decodeDrillBody(w, r)
	if !ok {
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	p.ID = generateID()
	p.Owner = username
	p.Created = now
	p.Updated = now
	drillsMu.Lock()
	drills[p.ID] = p
	err := saveDrillLocked(p)
	drillsMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill %s: %v", p.ID, err)
		jsonError(w, http.StatusInternalServerError, "could not save the problem")
		return
	}
	writeJSON(w, map[string]string{"id": p.ID})
}

// PUT /api/drills/{id} - the author edits their problem.
func handleDrillPut(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	p, _, ok := decodeDrillBody(w, r)
	if !ok {
		return
	}
	drillsMu.Lock()
	existing := drills[id]
	if existing == nil {
		drillsMu.Unlock()
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	if existing.Owner != username {
		drillsMu.Unlock()
		jsonError(w, http.StatusForbidden, "only the author can edit this problem")
		return
	}
	p.ID = id
	p.Owner = existing.Owner
	p.Created = existing.Created
	p.Updated = time.Now().UTC().Format(time.RFC3339)
	drills[id] = p
	err := saveDrillLocked(p)
	drillsMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill %s: %v", id, err)
		jsonError(w, http.StatusInternalServerError, "could not save the problem")
		return
	}
	writeJSON(w, map[string]any{"ok": true, "id": id})
}

// DELETE /api/drills/{id} - the author removes their problem.
func handleDrillDelete(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	drillsMu.Lock()
	p := drills[id]
	if p == nil {
		drillsMu.Unlock()
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	if p.Owner != username {
		drillsMu.Unlock()
		jsonError(w, http.StatusForbidden, "only the author can delete this problem")
		return
	}
	delete(drills, id)
	drillsMu.Unlock()
	if err := os.Remove(drillPath(id)); err != nil && !os.IsNotExist(err) {
		log.Printf("Could not remove drill file: %v", err)
	}
	writeJSON(w, map[string]bool{"ok": true})
}

// POST /api/drills/{id}/session - start practicing a problem.
func handleDrillSessionNew(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	p := lookupDrill(id)
	if p == nil {
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	drillsMu.Lock()
	snapshot := *p
	drillsMu.Unlock()
	s, err := newDrillSession(&snapshot, username)
	if err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}
	drillSessMu.Lock()
	gcDrillSessionsLocked()
	drillSess[s.ID] = s
	drillSessMu.Unlock()
	s.mu.Lock()
	st := s.stateLocked()
	s.mu.Unlock()
	writeJSON(w, st)
}

// GET /api/drill/session/{sid} - resume a practice session.
func handleDrillSessionGet(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	sid := r.PathValue("sid")
	if !validID(sid) {
		jsonError(w, http.StatusBadRequest, "invalid session ID")
		return
	}
	s := lookupDrillSession(sid, username)
	if s == nil {
		jsonError(w, http.StatusNotFound, "no such session")
		return
	}
	s.mu.Lock()
	st := s.stateLocked()
	s.mu.Unlock()
	writeJSON(w, st)
}

// POST /api/drill/session/{sid}/card - the hero's pick. Only the authored
// card is accepted; anything else answers 409 with the author's message for
// the correct card and plays nothing.
func handleDrillSessionCard(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	sid := r.PathValue("sid")
	if !validID(sid) {
		jsonError(w, http.StatusBadRequest, "invalid session ID")
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
	s := lookupDrillSession(sid, username)
	if s == nil {
		jsonError(w, http.StatusNotFound, "no such session")
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Done {
		writeJSON(w, s.stateLocked())
		return
	}
	seat := s.nextSeatLocked()
	if !s.heroControls(seat) {
		// a scripted seat should never be waiting; recover anyway
		s.advanceLocked()
		writeJSON(w, s.stateLocked())
		return
	}
	if !slices.Contains(s.legalCardsLocked(seat), card) {
		jsonError(w, http.StatusBadRequest, "illegal card: "+card)
		return
	}
	scripted := s.Problem.Plays[len(s.Plays)]
	if scripted.Card != card {
		s.Wrong++
		msg := strings.TrimSpace(scripted.Msg)
		if msg == "" {
			msg = defaultHeroMsg(scripted.Card)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		json.NewEncoder(w).Encode(map[string]any{
			"error": "not the card the author played",
			"hint":  msg,
			"wrong": s.Wrong,
		})
		return
	}
	s.applyLocked(scripted)
	s.advanceLocked()
	st := s.stateLocked()
	writeJSON(w, st)
}

// DELETE /api/drill/session/{sid} - abandon a practice session.
func handleDrillSessionDelete(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	sid := r.PathValue("sid")
	if !validID(sid) {
		jsonError(w, http.StatusBadRequest, "invalid session ID")
		return
	}
	s := lookupDrillSession(sid, username)
	if s == nil {
		jsonError(w, http.StatusNotFound, "no such session")
		return
	}
	drillSessMu.Lock()
	delete(drillSess, sid)
	drillSessMu.Unlock()
	writeJSON(w, map[string]bool{"ok": true})
}
