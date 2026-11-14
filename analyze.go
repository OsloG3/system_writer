package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ---- Hand analysis ----
//
// A full double-dummy breakdown of a played hand, card by card. For every card
// the analyzer reports whether it matched double-dummy best play, how many
// tricks it gave up and whether it flipped the contract between make and
// defeat; it also returns the deal's DD table, par and each side's makeable
// game/slam so a whole session of boards can be aggregated into stats.
//
// The DDS solver lives in the bidding-dt sidecar (POST /analyze, see
// play_server.py), so this is a thin, stateless proxy: the browser submits a
// deal -- pulled from its own play history or parsed from an uploaded PBN --
// with its auction and the cards played, and gets the per-card verdicts back.
// Anyone may analyze a submitted PBN; the play history a hand can be pulled
// from is itself login-gated (GET /api/play/stats). Bulk analysis is driven
// client-side: the page analyzes one board per request and aggregates, so a
// long session streams with a progress bar instead of one huge call.

// analyzeBoardBody is what the browser posts: a deal, its auction and the play.
// Plays carry the same shape as a stored PlayRecord.Plays (seat + "AS" card).
// ResultTricks optionally carries the declarer's actual trick count for a hand
// that was claimed / stopped short, so the score and play-vs-DD IMPs still work.
type analyzeBoardBody struct {
	Hands        [4]string  `json:"hands"`
	Dealer       int        `json:"dealer"`
	Vuln         int        `json:"vuln"`
	Calls        []string   `json:"calls"`
	Plays        []PlayCard `json:"plays"`
	ResultTricks *int       `json:"resultTricks"`
}

// POST /api/analyze/board - double-dummy analysis of one submitted hand.
func handleAnalyzeBoard(w http.ResponseWriter, r *http.Request) {
	limitBody(w, r, maxBodyAuth)
	var body analyzeBoardBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	for s := 0; s < 4; s++ {
		if body.Hands[s] == "" {
			jsonError(w, http.StatusBadRequest, "all four hands are required")
			return
		}
	}
	if body.Dealer < 0 || body.Dealer > 3 || body.Vuln < 0 || body.Vuln > 3 {
		jsonError(w, http.StatusBadRequest, "dealer and vuln must be 0..3")
		return
	}
	if len(body.Plays) > 52 {
		jsonError(w, http.StatusBadRequest, "a hand has at most 52 plays")
		return
	}
	plays := make([][2]any, 0, len(body.Plays))
	for _, pc := range body.Plays {
		card := normalizeCard(pc.Card)
		if card == "" || pc.Seat < 0 || pc.Seat > 3 {
			jsonError(w, http.StatusBadRequest, "invalid play")
			return
		}
		plays = append(plays, [2]any{pc.Seat, card})
	}
	req := map[string]any{
		"hands":  body.Hands,
		"dealer": body.Dealer,
		"vuln":   body.Vuln,
		"calls":  callsOrEmpty(body.Calls),
		"plays":  plays,
	}
	if body.ResultTricks != nil {
		if t := *body.ResultTricks; t >= 0 && t <= 13 {
			req["result_tricks"] = t
		}
	}
	var out map[string]any
	if err := botPost("/analyze", req, &out, botScoreTimeout); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	writeJSON(w, out)
}

// ---- Shared analyses ----
//
// An analysis lives entirely in the browser (boards come from the user's play
// history or a PBN), so sharing one needs a server-side snapshot: the boards
// plus their double-dummy results are stored under data/analysis/<id>.json and
// become reachable by anyone through /analyze?share=<id>. Creating a snapshot
// requires login; reading one does not -- like a system tree, the link is the
// capability (random 16-hex ID, never listed).

const maxBodyAnalysis = 8 << 20 // share POST: boards + full DD results
const maxShareBoards = 128      // one shared set is at most a full session

func analysisDir() string           { return filepath.Join(dataDir, "analysis") }
func analysisPath(id string) string { return filepath.Join(analysisDir(), id+".json") }

// sharedBoard is the slice of a client-side board a viewer needs: the deal,
// auction, play and display labels. Client-only fields (history timestamps,
// IMPs, par scores) are deliberately not carried into a share.
type sharedBoard struct {
	Title        string     `json:"title,omitempty"`
	Hands        [4]string  `json:"hands"`
	Dealer       int        `json:"dealer"`
	Vuln         int        `json:"vuln"`
	Calls        []string   `json:"calls"`
	Plays        []PlayCard `json:"plays"`
	ResultTricks *int       `json:"resultTricks"`
	Names        [4]string  `json:"names"`
	IdxLabel     *int       `json:"idxLabel,omitempty"`
	ContractStr  string     `json:"contractStr,omitempty"`
}

// SharedAnalysis is one stored snapshot. Results runs parallel to Boards:
// entry i is the raw double-dummy JSON for board i, or null when that board
// failed to analyze (a bulk set keeps going past a failed board).
type SharedAnalysis struct {
	ID      string            `json:"id"`
	Kind    string            `json:"kind"` // "single" | "bulk"
	Owner   string            `json:"owner,omitempty"`
	Created string            `json:"created"`
	Boards  []sharedBoard     `json:"boards"`
	Results []json.RawMessage `json:"results"`
}

// truncateShareText trims a display string carried into a share snapshot;
// these end up rendered to other users, so they are capped server-side.
func truncateShareText(s string, maxRunes int) string {
	s = strings.TrimSpace(s)
	r := []rune(s)
	if len(r) <= maxRunes {
		return s
	}
	return strings.TrimSpace(string(r[:maxRunes]))
}

// POST /api/analyze/share - store a snapshot of a single-hand or bulk
// analysis so it can be opened by anyone through /analyze?share=<id>.
func handleAnalyzeShareCreate(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	limitBody(w, r, maxBodyAnalysis)
	var body struct {
		Kind    string            `json:"kind"`
		Boards  []sharedBoard     `json:"boards"`
		Results []json.RawMessage `json:"results"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if body.Kind != "single" && body.Kind != "bulk" {
		jsonError(w, http.StatusBadRequest, `kind must be "single" or "bulk"`)
		return
	}
	if len(body.Boards) == 0 || len(body.Boards) > maxShareBoards {
		jsonError(w, http.StatusBadRequest, fmt.Sprintf("a share needs 1-%d boards", maxShareBoards))
		return
	}
	if len(body.Results) != len(body.Boards) {
		jsonError(w, http.StatusBadRequest, "results must line up with boards")
		return
	}
	for i := range body.Boards {
		b := &body.Boards[i]
		for s := 0; s < 4; s++ {
			if strings.TrimSpace(b.Hands[s]) == "" {
				jsonError(w, http.StatusBadRequest, "every board needs all four hands")
				return
			}
		}
		if b.Dealer < 0 || b.Dealer > 3 || b.Vuln < 0 || b.Vuln > 3 {
			jsonError(w, http.StatusBadRequest, "dealer and vuln must be 0..3")
			return
		}
		if len(b.Plays) > 52 {
			jsonError(w, http.StatusBadRequest, "a hand has at most 52 plays")
			return
		}
		plays := make([]PlayCard, 0, len(b.Plays))
		for _, pc := range b.Plays {
			card := normalizeCard(pc.Card)
			if card == "" || pc.Seat < 0 || pc.Seat > 3 {
				jsonError(w, http.StatusBadRequest, "invalid play")
				return
			}
			plays = append(plays, PlayCard{Seat: pc.Seat, Card: card})
		}
		b.Plays = plays
		b.Calls = callsOrEmpty(b.Calls)
		b.Title = truncateShareText(b.Title, 120)
		b.ContractStr = truncateShareText(b.ContractStr, 80)
		for s := range b.Names {
			b.Names[s] = truncateShareText(b.Names[s], 60)
		}
		if len(body.Results[i]) == 0 || string(body.Results[i]) == "null" {
			body.Results[i] = json.RawMessage("null")
			continue
		}
		var probe map[string]any
		if err := json.Unmarshal(body.Results[i], &probe); err != nil {
			jsonError(w, http.StatusBadRequest, "each result must be an analysis object or null")
			return
		}
	}
	// A single-hand share is exactly one board with its result; a bulk set
	// may carry nulls for boards that failed to analyze.
	if body.Kind == "single" && (len(body.Boards) != 1 || string(body.Results[0]) == "null") {
		jsonError(w, http.StatusBadRequest, "a single share needs one board and its result")
		return
	}
	shared := SharedAnalysis{
		ID:      generateID(),
		Kind:    body.Kind,
		Owner:   username,
		Created: time.Now().UTC().Format(time.RFC3339),
		Boards:  body.Boards,
		Results: body.Results,
	}
	if err := os.MkdirAll(analysisDir(), 0o755); err != nil {
		jsonError(w, http.StatusInternalServerError, "could not save the analysis")
		return
	}
	if err := writeJSONFile(analysisPath(shared.ID), &shared); err != nil {
		log.Printf("Could not save shared analysis %s: %v", shared.ID, err)
		jsonError(w, http.StatusInternalServerError, "could not save the analysis")
		return
	}
	writeJSON(w, map[string]any{"id": shared.ID, "url": "/analyze?share=" + shared.ID})
}

// GET /api/analyze/share/{id} - public read of a shared analysis snapshot.
func handleAnalyzeShareGet(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid analysis ID")
		return
	}
	data, err := os.ReadFile(analysisPath(id))
	if err != nil {
		jsonError(w, http.StatusNotFound, "no such shared analysis")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}
