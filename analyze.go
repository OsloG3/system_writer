package main

import (
	"encoding/json"
	"net/http"
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
type analyzeBoardBody struct {
	Hands  [4]string  `json:"hands"`
	Dealer int        `json:"dealer"`
	Vuln   int        `json:"vuln"`
	Calls  []string   `json:"calls"`
	Plays  []PlayCard `json:"plays"`
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
	var out map[string]any
	if err := botPost("/analyze", req, &out, botScoreTimeout); err != nil {
		jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		return
	}
	writeJSON(w, out)
}
