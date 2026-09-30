package main

import (
	"crypto/rand"
	"encoding/json"
	"errors"
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

// ---- Partner tables: two humans (N/S) vs the East/West bots ----
//
// The host sits South, the partner sits North; East and West are the same
// bidding-dt bots as in solo play. Every table is persisted to disk
// (data/game_<id>.json) so a game keeps its state while both players are
// disconnected and survives server restarts. A player can keep any number of
// tables open with the same partner and resume each one from the play lobby.

const (
	teamSouthSeat    = 2 // host
	teamNorthSeat    = 0 // partner
	joinCodeAlphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
	joinCodeLen      = 6
)

var (
	teamMu    sync.Mutex
	teamGames = map[string]*TeamGame{} // keyed by game ID
)

// TeamBoard is one dealt board inside a table (auction + DD result).
type TeamBoard struct {
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
	mu      sync.Mutex   `json:"-"`
	ID      string       `json:"id"`
	Code    string       `json:"code"` // short join code shared with the partner
	Host    string       `json:"host"` // username sitting South
	Partner string       `json:"partner"`
	Boards  []*TeamBoard `json:"boards"`
	Created time.Time    `json:"created"`
	Updated time.Time    `json:"updated"`
}

// Standard 16-board duplicate rotation: the dealer moves N,E,S,W while the
// vulnerability follows the WBF cycle (board 1: N, none; board 2: E, N-S; ...).
var teamVulnCycle = [16]int{0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2}

func teamRotation(boardNo int) (dealer, vuln int) {
	return boardNo % 4, teamVulnCycle[boardNo%16]
}

func initTeam() {
	files, err := filepath.Glob(filepath.Join(dataDir, "game_*.json"))
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
		g.ID = strings.TrimPrefix(strings.TrimSuffix(filepath.Base(f), ".json"), "game_")
		if !validID(g.ID) {
			continue
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
	return filepath.Join(dataDir, "game_"+id+".json")
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

// lastLocked returns the current board (the last one), or nil before the
// first deal. Callers must hold g.mu.
func (g *TeamGame) lastLocked() *TeamBoard {
	if len(g.Boards) == 0 {
		return nil
	}
	return g.Boards[len(g.Boards)-1]
}

func auctionFinished(calls []string) bool {
	n := len(calls)
	return n >= 4 && calls[n-1] == "P" && calls[n-2] == "P" && calls[n-3] == "P"
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
	b.Legal = nil
	b.LegalLen = -1
	g.Updated = time.Now()
	return nil
}

// stateLocked renders the client-visible table for one member. Callers must
// hold g.mu.
func (g *TeamGame) stateLocked(username string) map[string]any {
	seat := g.seatLocked(username)
	st := map[string]any{
		"mode":              "team",
		"id":                g.ID,
		"code":              g.Code,
		"host":              g.Host,
		"partner":           g.Partner,
		"yourSeat":          seat,
		"humanSeat":         seat,
		"bots":              botLabels(),
		"seatNames":         map[string]string{"0": g.Partner, "2": g.Host},
		"waitingForPartner": g.Partner == "",
		"totalBoards":       len(g.Boards),
		"stats":             g.statsLocked(),
	}
	b := g.lastLocked()
	if b == nil {
		st["noBoard"] = true
		return st
	}
	st["boardNo"] = len(g.Boards)
	st["dealer"] = b.Dealer
	st["vuln"] = b.Vuln
	st["calls"] = callsOrEmpty(b.Calls)
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
		legal, err := g.legalLocked(b)
		if err == nil {
			st["legal"] = legal
		}
	}
	return st
}

// ---- Handlers ----

// POST /api/play/team/new - opens a table and returns the join code
func handleTeamNew(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	limitBody(w, r, maxBodyAuth)
	g := &TeamGame{
		ID:      generateID(),
		Code:    newJoinCode(),
		Host:    username,
		Boards:  []*TeamBoard{},
		Created: time.Now(),
		Updated: time.Now(),
	}
	teamMu.Lock()
	teamGames[g.ID] = g
	teamMu.Unlock()
	if err := saveTeamGameLocked(g); err != nil {
		log.Printf("Could not persist team game: %v", err)
	}
	g.mu.Lock()
	st := g.stateLocked(username)
	g.mu.Unlock()
	writeJSON(w, st)
}

// POST /api/play/team/join - partner joins an open table by its code
func handleTeamJoin(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Code string `json:"code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	code := strings.ToUpper(strings.TrimSpace(body.Code))
	if code == "" {
		jsonError(w, http.StatusBadRequest, "missing join code")
		return
	}
	teamMu.Lock()
	var g *TeamGame
	for _, x := range teamGames {
		if x.Code == code {
			g = x
			break
		}
	}
	teamMu.Unlock()
	if g == nil {
		jsonError(w, http.StatusNotFound, "no table with that code")
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.Partner == "" {
		g.Partner = username
		g.Updated = time.Now()
	} else if g.Partner != username {
		jsonError(w, http.StatusConflict, "that table already has a partner")
		return
	}
	if err := saveTeamGameLocked(g); err != nil {
		log.Printf("Could not persist team game: %v", err)
	}
	writeJSON(w, g.stateLocked(username))
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
		ID                string    `json:"id"`
		Code              string    `json:"code"`
		Host              string    `json:"host"`
		Partner           string    `json:"partner"`
		YourSeat          int       `json:"yourSeat"`
		Boards            int       `json:"boards"`
		ImpsTotal         float64   `json:"impsTotal"`
		InProgress        bool      `json:"inProgress"`
		YourTurn          bool      `json:"yourTurn"`
		WaitingForPartner bool      `json:"waitingForPartner"`
		Updated           time.Time `json:"updated"`
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
			ID: g.ID, Code: g.Code, Host: g.Host, Partner: g.Partner,
			YourSeat: seat, WaitingForPartner: g.Partner == "",
			Updated: g.Updated,
		}
		for _, b := range g.Boards {
			if b.Result != nil {
				s.Boards++
				s.ImpsTotal += b.Result.Imps
			}
		}
		if b := g.lastLocked(); b != nil && b.Result == nil {
			s.InProgress = true
			s.YourTurn = (b.Dealer+len(b.Calls))%4 == seat
		}
		g.mu.Unlock()
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Updated.After(out[j].Updated) })
	writeJSON(w, map[string]any{"games": out})
}

// GET /api/play/team/{id} - table state for the polling client; also nudges
// stalled bot turns after a sidecar hiccup
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
	if b := g.lastLocked(); b != nil && b.Result == nil {
		before := len(b.Calls)
		hadResult := b.Result != nil
		err := g.advanceLocked(b)
		if len(b.Calls) != before || (!hadResult && b.Result != nil) {
			if serr := saveTeamGameLocked(g); serr != nil {
				log.Printf("Could not persist team game: %v", serr)
			}
		}
		if err != nil {
			writeJSON(w, g.stateLocked(username)) // return what we have; client retries
			return
		}
	}
	writeJSON(w, g.stateLocked(username))
}

// POST /api/play/team/{id}/call - the human's call for their seat
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
	b := g.lastLocked()
	if b == nil || b.Result != nil {
		jsonError(w, http.StatusConflict, "no board in progress")
		return
	}
	if next := (b.Dealer + len(b.Calls)) % 4; next != seat {
		jsonError(w, http.StatusConflict, "not your turn")
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
	b.Calls = append(b.Calls, call)
	b.Legal = nil
	b.LegalLen = -1
	err = g.advanceLocked(b)
	if serr := saveTeamGameLocked(g); serr != nil {
		log.Printf("Could not persist team game: %v", serr)
	}
	if err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	writeJSON(w, g.stateLocked(username))
}

// POST /api/play/team/{id}/next - deal the next board (either member)
func handleTeamNext(w http.ResponseWriter, r *http.Request) {
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
	if g.Partner == "" {
		jsonError(w, http.StatusConflict, "waiting for a partner to join")
		return
	}
	if b := g.lastLocked(); b != nil && b.Result == nil {
		jsonError(w, http.StatusConflict, "the current board is still in progress")
		return
	}
	dealer, vuln := teamRotation(len(g.Boards))
	var deal struct {
		Hands  [4]string `json:"hands"`
		Dealer int       `json:"dealer"`
		Vuln   int       `json:"vuln"`
	}
	if err := botPost("/deal", map[string]any{"dealer": dealer, "vuln": vuln}, &deal, botTimeout); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	b := &TeamBoard{Hands: deal.Hands, Dealer: deal.Dealer, Vuln: deal.Vuln}
	g.Boards = append(g.Boards, b)
	if err := g.advanceLocked(b); err != nil {
		if serr := saveTeamGameLocked(g); serr != nil {
			log.Printf("Could not persist team game: %v", serr)
		}
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	if err := saveTeamGameLocked(g); err != nil {
		log.Printf("Could not persist team game: %v", err)
	}
	writeJSON(w, g.stateLocked(username))
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
		Dealer int         `json:"dealer"`
		Vuln   int         `json:"vuln"`
		Calls  []string    `json:"calls"`
		Hands  [4]string   `json:"hands"`
		Result *PlayResult `json:"result"`
	}
	boards := make([]historyBoard, 0, len(g.Boards))
	for i, b := range g.Boards {
		boards = append(boards, historyBoard{
			No: i + 1, Dealer: b.Dealer, Vuln: b.Vuln,
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

// newJoinCode returns an unused 6-character code (no I/L/O/0/1 confusion).
func newJoinCode() string {
	for {
		raw := make([]byte, joinCodeLen)
		rand.Read(raw)
		code := make([]byte, joinCodeLen)
		for i := range code {
			code[i] = joinCodeAlphabet[int(raw[i])%len(joinCodeAlphabet)]
		}
		s := string(code)
		teamMu.Lock()
		used := false
		for _, g := range teamGames {
			if g.Code == s {
				used = true
				break
			}
		}
		teamMu.Unlock()
		if !used {
			return s
		}
	}
}
