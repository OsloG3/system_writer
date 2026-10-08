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
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---- Partner tables: two humans (N/S) vs the East/West bots ----
//
// The host sits South, the partner sits North; East and West are the same
// bidding-dt bots as in solo play. A table is opened by naming the partner's
// account, so either player can find it in their table list and no link is
// needed. Every table is persisted to disk (data/game/<id>.json) so a game
// keeps its state while both players are disconnected and survives server
// restarts. A player can keep any number of tables open with the same partner
// and resume each one from the play lobby.
//
// A match is teamMatchLen boards, all dealt up front when either player
// starts it. Every board runs its own auction: each player may call on any
// board where it is their turn, switching between boards freely instead of
// playing them in order. A new match can be started once every board of the
// current one is finished; board numbering and the dealer/vulnerability
// rotation simply continue.

const (
	teamSouthSeat = 2 // host
	teamNorthSeat = 0 // partner
	teamMatchLen  = 8 // boards per match
)

var (
	teamMu    sync.Mutex
	teamGames = map[string]*TeamGame{} // keyed by game ID
)

// TeamBoard is one dealt board inside a table (auction + DD result).
type TeamBoard struct {
	Ts       string      `json:"ts,omitempty"` // RFC3339, set when the board was scored
	Dealer   int         `json:"dealer"`
	Vuln     int         `json:"vuln"`
	Hands    [4]string   `json:"hands"`
	Calls    []string    `json:"calls"`
	Result   *PlayResult `json:"result,omitempty"`
	Legal    []string    `json:"-"` // legal-call cache for the seat on turn
	LegalLen int         `json:"-"` // len(Calls) the cache was built for
}

// TeamGame is a persistent two-player table.
type TeamGame struct {
	mu         sync.Mutex   `json:"-"`
	ID         string       `json:"id"`
	Host       string       `json:"host"`    // username sitting South
	Partner    string       `json:"partner"` // username sitting North
	Boards     []*TeamBoard `json:"boards"`
	MatchStart int          `json:"matchStart"` // index of the first board of the current match
	Created    time.Time    `json:"created"`
	Updated    time.Time    `json:"updated"`
}

// Standard 16-board duplicate rotation: the dealer moves N,E,S,W while the
// vulnerability follows the WBF cycle (board 1: N, none; board 2: E, N-S; ...).
var teamVulnCycle = [16]int{0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2}

func teamRotation(boardNo int) (dealer, vuln int) {
	return boardNo % 4, teamVulnCycle[boardNo%16]
}

func initTeam() {
	files, err := filepath.Glob(filepath.Join(gameDir(), "*.json"))
	if err != nil {
		return
	}
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var g TeamGame
		if err := json.Unmarshal(data, &g); err != nil {
			log.Printf("Could not parse %s: %v", f, err)
			continue
		}
		g.ID = strings.TrimSuffix(filepath.Base(f), ".json")
		if !validID(g.ID) {
			continue // play_stats.json and any other non-game files
		}
		if g.Boards == nil {
			g.Boards = []*TeamBoard{}
		}
		teamGames[g.ID] = &g
	}
	if len(teamGames) > 0 {
		log.Printf("Loaded %d team game(s)", len(teamGames))
	}
}

func teamGamePath(id string) string {
	return filepath.Join(gameDir(), id+".json")
}

// saveTeamGameLocked persists a game. Callers must hold g.mu.
func saveTeamGameLocked(g *TeamGame) error {
	return writeJSONFile(teamGamePath(g.ID), g)
}

// lookupTeam returns the game or nil; callers then take g.mu.
func lookupTeam(id string) *TeamGame {
	teamMu.Lock()
	defer teamMu.Unlock()
	return teamGames[id]
}

// seatLocked returns the member's seat (S=2 host, N=0 partner) or -1.
// Callers must hold g.mu.
func (g *TeamGame) seatLocked(username string) int {
	if username != "" && username == g.Host {
		return teamSouthSeat
	}
	if username != "" && username == g.Partner {
		return teamNorthSeat
	}
	return -1
}

func auctionFinished(calls []string) bool {
	n := len(calls)
	return n >= 4 && calls[n-1] == "P" && calls[n-2] == "P" && calls[n-3] == "P"
}

// waitingLocked maps each human seat to the 1-based numbers of the current
// match's boards that await that seat's bid. Callers must hold g.mu.
func (g *TeamGame) waitingLocked() map[int][]int {
	m := make(map[int][]int)
	start := g.matchStartIdx()
	for i := start; i < len(g.Boards); i++ {
		b := g.Boards[i]
		if b.Result != nil {
			continue
		}
		if next := (b.Dealer + len(b.Calls)) % 4; next == teamSouthSeat || next == teamNorthSeat {
			m[next] = append(m[next], i+1)
		}
	}
	return m
}

// newTurns returns, per human seat, the boards gained between two
// waitingLocked snapshots.
func newTurns(before, after map[int][]int) map[int][]int {
	out := make(map[int][]int)
	for seat, boards := range after {
		var fresh []int
		for _, no := range boards {
			if !slices.Contains(before[seat], no) {
				fresh = append(fresh, no)
			}
		}
		if len(fresh) > 0 {
			out[seat] = fresh
		}
	}
	return out
}

// pushNewTurns sends Web Push alerts for the boards that became a member's
// turn between two waitingLocked snapshots -- the only delivery path for
// players whose page is closed. One combined push per member.
func (g *TeamGame) pushNewTurns(before, after map[int][]int) {
	for seat, fresh := range newTurns(before, after) {
		user := g.Partner
		if seat == teamSouthSeat {
			user = g.Host
		}
		if user == "" {
			continue
		}
		sendTurnPush(user, g.ID, fresh)
	}
}

// advanceAllLocked runs every board of the current match whose bots owe
// calls. Callers must hold g.mu.
func (g *TeamGame) advanceAllLocked() {
	for _, b := range g.Boards[g.matchStartIdx():] {
		if b.Result != nil {
			continue
		}
		if err := g.advanceLocked(b); err != nil {
			log.Printf("team board advance: %v", err)
		}
	}
}

// statsLocked summarizes the table so the shared scoreboard renders the same
// way as solo play. Callers must hold g.mu.
func (g *TeamGame) statsLocked() map[string]any {
	boards, total := 0, 0.0
	for _, b := range g.Boards {
		if b.Result != nil {
			boards++
			total += b.Result.Imps
		}
	}
	avg := 0.0
	if boards > 0 {
		avg = total / float64(boards)
	}
	return map[string]any{
		"boards": boards, "impsTotal": total, "avgImps": avg,
		"recent": []any{},
	}
}

// legalLocked fetches (and caches) the legal calls for the board. Callers
// must hold g.mu.
func (g *TeamGame) legalLocked(b *TeamBoard) ([]string, error) {
	if b.Legal != nil && b.LegalLen == len(b.Calls) {
		return b.Legal, nil
	}
	var out struct {
		Seat  int      `json:"seat"`
		Over  bool     `json:"over"`
		Legal []string `json:"legal"`
	}
	body := map[string]any{
		"dealer": b.Dealer, "vuln": b.Vuln, "calls": callsOrEmpty(b.Calls),
	}
	if err := botPost("/legal", body, &out, botTimeout); err != nil {
		return nil, err
	}
	b.Legal = out.Legal
	b.LegalLen = len(b.Calls)
	return b.Legal, nil
}

// advanceLocked runs the bots until a human must act, the auction ends or the
// model call cap is hit, then scores. Callers must hold g.mu.
func (g *TeamGame) advanceLocked(b *TeamBoard) error {
	for b.Result == nil && !auctionFinished(b.Calls) && len(b.Calls) < maxAuctionCalls {
		seat := (b.Dealer + len(b.Calls)) % 4
		if seat == teamSouthSeat || seat == teamNorthSeat {
			return nil // wait for a human call
		}
		body := map[string]any{
			"hands":  b.Hands,
			"dealer": b.Dealer,
			"vuln":   b.Vuln,
			"calls":  callsOrEmpty(b.Calls),
		}
		if m := botModels[seat]; m != "" {
			body["model"] = m
		}
		var out struct {
			Seat int    `json:"seat"`
			Call string `json:"call"`
		}
		if err := botPost("/bid", body, &out, botTimeout); err != nil {
			return err
		}
		if out.Seat != seat || normalizeCall(out.Call) == "" {
			return errors.New("sidecar returned an inconsistent bid")
		}
		b.Calls = append(b.Calls, out.Call)
		b.Legal = nil
		b.LegalLen = -1
	}
	if b.Result == nil && (auctionFinished(b.Calls) || len(b.Calls) >= maxAuctionCalls) {
		return g.scoreLocked(b)
	}
	return nil
}

// scoreLocked asks the sidecar for the double-dummy par result. Callers must
// hold g.mu.
func (g *TeamGame) scoreLocked(b *TeamBoard) error {
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
	if b.Ts == "" {
		b.Ts = time.Now().UTC().Format(time.RFC3339)
	}
	b.Legal = nil
	b.LegalLen = -1
	g.Updated = time.Now()
	return nil
}

// dealMatchLocked deals a fresh teamMatchLen-board match and runs the bots'
// openings on every board whose dealer is E/W. The whole match is dealt before
// any board is committed so a sidecar failure mid-deal cannot leave a partial
// match behind. Callers must hold g.mu.
func (g *TeamGame) dealMatchLocked() error {
	start := len(g.Boards)
	type dealT struct {
		Hands  [4]string `json:"hands"`
		Dealer int       `json:"dealer"`
		Vuln   int       `json:"vuln"`
	}
	deals := make([]dealT, 0, teamMatchLen)
	for i := 0; i < teamMatchLen; i++ {
		dealer, vuln := teamRotation(start + i)
		var d dealT
		if err := botPost("/deal", map[string]any{"dealer": dealer, "vuln": vuln}, &d, botTimeout); err != nil {
			return err
		}
		deals = append(deals, d)
	}
	g.MatchStart = start
	for _, d := range deals {
		b := &TeamBoard{Hands: d.Hands, Dealer: d.Dealer, Vuln: d.Vuln}
		g.Boards = append(g.Boards, b)
		// Bots open every board whose dealer is E/W; boards dealt to a human
		// dealer simply wait. A failure here leaves the board waiting on a
		// bot turn, which polling recovers.
		if err := g.advanceLocked(b); err != nil {
			log.Printf("team deal board advance: %v", err)
		}
	}
	g.Updated = time.Now()
	return nil
}

// boardAtLocked returns the board with the given 1-based number, or nil.
// Callers must hold g.mu.
func (g *TeamGame) boardAtLocked(no int) *TeamBoard {
	if no < 1 || no > len(g.Boards) {
		return nil
	}
	return g.Boards[no-1]
}

// matchStartIdx is the index of the first board of the current match.
func (g *TeamGame) matchStartIdx() int { return min(g.MatchStart, len(g.Boards)) }

// stateLocked renders the client-visible table for one member: identity
// fields, a summary of every board of the current match (for the board
// switcher), and the full state of the selected board flattened on top.
// boardNo is a 1-based board number; 0 (or out of match) lets the server
// pick a default: the first board waiting for this player, else the first
// unfinished board, else the last board of the match. Callers must hold
// g.mu.
func (g *TeamGame) stateLocked(username string, boardNo int) map[string]any {
	seat := g.seatLocked(username)
	st := map[string]any{
		"mode":        "team",
		"id":          g.ID,
		"host":        g.Host,
		"partner":     g.Partner,
		"yourSeat":    seat,
		"humanSeat":   seat,
		"bots":        botLabels(),
		"seatNames":   map[string]string{"0": g.Partner, "2": g.Host},
		"totalBoards": len(g.Boards),
		"stats":       g.statsLocked(),
	}
	start := g.matchStartIdx()
	summaries := make([]map[string]any, 0, len(g.Boards)-start)
	matchDone, matchImps := len(g.Boards) > start, 0.0
	firstYour, firstOpen, last := 0, 0, 0
	for i, b := range g.Boards[start:] {
		no := start + i + 1
		sum := map[string]any{"no": no, "dealer": b.Dealer, "vuln": b.Vuln}
		if b.Result != nil {
			sum["done"] = true
			sum["imps"] = b.Result.Imps
			sum["contract"] = b.Result.Contract.String()
			matchImps += b.Result.Imps
		} else {
			sum["done"] = false
			matchDone = false
			next := (b.Dealer + len(b.Calls)) % 4
			yt := seat >= 0 && next == seat
			sum["nextSeat"] = next
			sum["yourTurn"] = yt
			if firstYour == 0 && yt {
				firstYour = no
			}
			if firstOpen == 0 {
				firstOpen = no
			}
		}
		last = no
		summaries = append(summaries, sum)
	}
	st["boards"] = summaries
	st["matchDone"] = matchDone
	st["matchImps"] = matchImps

	sel := boardNo
	if sel <= start || sel > len(g.Boards) {
		switch {
		case firstYour > 0:
			sel = firstYour
		case firstOpen > 0:
			sel = firstOpen
		default:
			sel = last
		}
	}
	st["boardSel"] = sel
	if sel == 0 {
		st["noBoard"] = true
		return st
	}
	b := g.Boards[sel-1]
	st["boardNo"] = sel
	st["dealer"] = b.Dealer
	st["vuln"] = b.Vuln
	st["calls"] = callsOrEmpty(b.Calls)
	// Alerts for every East/West (bot) bid of the selected board, parallel
	// to calls; omitted until the self-play tree finishes loading.
	if t := alertTreeReady(); t != nil {
		if a := t.callAlerts(b.Dealer, b.Calls, 1<<teamSouthSeat|1<<teamNorthSeat); a != nil {
			st["alerts"] = a
		}
	}
	next := (b.Dealer + len(b.Calls)) % 4
	st["nextSeat"] = next
	st["done"] = b.Result != nil
	if b.Result != nil {
		st["yourTurn"] = false
		st["hands"] = b.Hands
		st["result"] = b.Result
		return st
	}
	yourTurn := seat >= 0 && next == seat
	st["yourTurn"] = yourTurn
	if seat >= 0 {
		st["hand"] = b.Hands[seat]
	}
	if yourTurn {
		if legal, err := g.legalLocked(b); err == nil {
			st["legal"] = legal
			if t := alertTreeReady(); t != nil {
				if a := t.optionAlerts(b.Calls, legal); a != nil {
					st["optionAlerts"] = a
				}
			}
		}
	}
	return st
}

// ---- Handlers ----

// POST /api/play/team/new - opens a table with a named partner account
func handleTeamNew(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Partner string `json:"partner"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	partner := strings.ToLower(strings.TrimSpace(body.Partner))
	if partner == "" {
		jsonError(w, http.StatusBadRequest, "enter your partner's username")
		return
	}
	if partner == username {
		jsonError(w, http.StatusBadRequest, "pick a different partner")
		return
	}
	if !userExists(partner) {
		jsonError(w, http.StatusNotFound, "no account named "+partner)
		return
	}
	g := &TeamGame{
		ID:      generateID(),
		Host:    username,
		Partner: partner,
		Boards:  []*TeamBoard{},
		Created: time.Now(),
		Updated: time.Now(),
	}
	teamMu.Lock()
	teamGames[g.ID] = g
	teamMu.Unlock()
	g.mu.Lock()
	if err := saveTeamGameLocked(g); err != nil {
		log.Printf("Could not persist team game: %v", err)
	}
	// Deal the first match right away so both members see the boards as soon
	// as the table exists, without either having to start it. A sidecar
	// failure leaves an empty table that can still be started manually.
	if err := g.dealMatchLocked(); err != nil {
		log.Printf("team new: dealing first match: %v", err)
	} else if err := saveTeamGameLocked(g); err != nil {
		log.Printf("Could not persist team game: %v", err)
	}
	g.pushNewTurns(nil, g.waitingLocked())
	st := g.stateLocked(username, 0)
	g.mu.Unlock()
	writeJSON(w, st)
}

// GET /api/play/team/list - the tables this user is part of
func handleTeamList(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	teamMu.Lock()
	all := make([]*TeamGame, 0, len(teamGames))
	for _, g := range teamGames {
		all = append(all, g)
	}
	teamMu.Unlock()

	type gameSummary struct {
		ID         string    `json:"id"`
		Host       string    `json:"host"`
		Partner    string    `json:"partner"`
		YourSeat   int       `json:"yourSeat"`
		Boards     int       `json:"boards"`
		ImpsTotal  float64   `json:"impsTotal"`
		InProgress bool      `json:"inProgress"`
		YourTurn   bool      `json:"yourTurn"`
		Updated    time.Time `json:"updated"`
	}
	out := []gameSummary{}
	for _, g := range all {
		g.mu.Lock()
		seat := g.seatLocked(username)
		if seat < 0 {
			g.mu.Unlock()
			continue
		}
		s := gameSummary{
			ID: g.ID, Host: g.Host, Partner: g.Partner,
			YourSeat: seat, Updated: g.Updated,
		}
		for _, b := range g.Boards {
			if b.Result != nil {
				s.Boards++
				s.ImpsTotal += b.Result.Imps
				continue
			}
			s.InProgress = true
			if (b.Dealer+len(b.Calls))%4 == seat {
				s.YourTurn = true
			}
		}
		g.mu.Unlock()
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Updated.After(out[j].Updated) })
	writeJSON(w, map[string]any{"games": out})
}

// GET /api/play/team/{id}?board=N - table state for the polling client with
// board N selected (0/absent = server default). Also advances every board
// whose bots owe calls, recovering stalled turns after a sidecar hiccup.
func handleTeamGet(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid game ID")
		return
	}
	g := lookupTeam(id)
	if g == nil {
		jsonError(w, http.StatusNotFound, "no such table")
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.seatLocked(username) < 0 {
		jsonError(w, http.StatusForbidden, "not part of this table")
		return
	}
	boardNo, _ := strconv.Atoi(r.URL.Query().Get("board"))
	waitBefore := g.waitingLocked()
	changed := false
	for _, b := range g.Boards[g.matchStartIdx():] {
		if b.Result != nil {
			continue
		}
		before := len(b.Calls)
		if err := g.advanceLocked(b); err != nil {
			log.Printf("team board advance: %v", err)
		}
		if len(b.Calls) != before || b.Result != nil {
			changed = true
		}
	}
	if changed {
		if serr := saveTeamGameLocked(g); serr != nil {
			log.Printf("Could not persist team game: %v", serr)
		}
	}
	g.pushNewTurns(waitBefore, g.waitingLocked())
	writeJSON(w, g.stateLocked(username, boardNo))
}

// POST /api/play/team/{id}/call - the human's call on one board of the match
func handleTeamCall(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid game ID")
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Call  string `json:"call"`
		Board int    `json:"board"` // 1-based board number
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
	g := lookupTeam(id)
	if g == nil {
		jsonError(w, http.StatusNotFound, "no such table")
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	seat := g.seatLocked(username)
	if seat < 0 {
		jsonError(w, http.StatusForbidden, "not part of this table")
		return
	}
	b := g.boardAtLocked(body.Board)
	if b == nil || b.Result != nil {
		jsonError(w, http.StatusConflict, "no such board, or it is finished")
		return
	}
	if next := (b.Dealer + len(b.Calls)) % 4; next != seat {
		jsonError(w, http.StatusConflict, fmt.Sprintf("not your turn on board %d", body.Board))
		return
	}
	legal, err := g.legalLocked(b)
	if err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	if !slices.Contains(legal, call) {
		jsonError(w, http.StatusBadRequest, "illegal call: "+call)
		return
	}
	waitBefore := g.waitingLocked()
	b.Calls = append(b.Calls, call)
	b.Legal = nil
	b.LegalLen = -1
	g.Updated = time.Now()
	err = g.advanceLocked(b)
	// run the match's other stalled boards too, so a member whose page is
	// closed still learns via Web Push that a board awaits them
	g.advanceAllLocked()
	if serr := saveTeamGameLocked(g); serr != nil {
		log.Printf("Could not persist team game: %v", serr)
	}
	g.pushNewTurns(waitBefore, g.waitingLocked())
	if err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	writeJSON(w, g.stateLocked(username, body.Board))
}

// POST /api/play/team/{id}/start - deal a fresh teamMatchLen-board match (either
// member). All boards are dealt up front, each auction running independently;
// the players bid them in any order they like.
func handleTeamStart(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid game ID")
		return
	}
	g := lookupTeam(id)
	if g == nil {
		jsonError(w, http.StatusNotFound, "no such table")
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.seatLocked(username) < 0 {
		jsonError(w, http.StatusForbidden, "not part of this table")
		return
	}
	for _, b := range g.Boards {
		if b.Result == nil {
			jsonError(w, http.StatusConflict, "the current match is still in progress")
			return
		}
	}
	// Deal the whole match first; only commit once every board is in hand so
	// a sidecar failure mid-deal cannot leave a partial match behind.
	if err := g.dealMatchLocked(); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	if err := saveTeamGameLocked(g); err != nil {
		log.Printf("Could not persist team game: %v", err)
	}
	g.pushNewTurns(nil, g.waitingLocked())
	writeJSON(w, g.stateLocked(username, 0))
}

// GET /api/play/team/{id}/history - every board of the table with hands and
// the full auction (for the review overlay)
func handleTeamHistory(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid game ID")
		return
	}
	g := lookupTeam(id)
	if g == nil {
		jsonError(w, http.StatusNotFound, "no such table")
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.seatLocked(username) < 0 {
		jsonError(w, http.StatusForbidden, "not part of this table")
		return
	}
	type historyBoard struct {
		No     int         `json:"no"`
		Match  int         `json:"match"` // 1-based match number (teamMatchLen boards each)
		Dealer int         `json:"dealer"`
		Vuln   int         `json:"vuln"`
		Calls  []string    `json:"calls"`
		Hands  [4]string   `json:"hands"`
		Result *PlayResult `json:"result"`
	}
	boards := make([]historyBoard, 0, len(g.Boards))
	for i, b := range g.Boards {
		boards = append(boards, historyBoard{
			No: i + 1, Match: i/teamMatchLen + 1, Dealer: b.Dealer, Vuln: b.Vuln,
			Calls: callsOrEmpty(b.Calls), Hands: b.Hands, Result: b.Result,
		})
	}
	writeJSON(w, map[string]any{"boards": boards})
}

// DELETE /api/play/team/{id} - either member closes the table for good
func handleTeamDelete(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid game ID")
		return
	}
	g := lookupTeam(id)
	if g == nil {
		jsonError(w, http.StatusNotFound, "no such table")
		return
	}
	g.mu.Lock()
	if g.seatLocked(username) < 0 {
		g.mu.Unlock()
		jsonError(w, http.StatusForbidden, "not part of this table")
		return
	}
	g.mu.Unlock()
	teamMu.Lock()
	delete(teamGames, id)
	teamMu.Unlock()
	if err := os.Remove(teamGamePath(id)); err != nil && !os.IsNotExist(err) {
		log.Printf("Could not remove team game file: %v", err)
	}
	writeJSON(w, map[string]bool{"ok": true})
}
