package main

import (
	"math"
	"net/http"
	"sort"
	"time"
)

// ---- Rankings ----
//
// Three independent leaderboards, all scored the same way: the average IMPs
// per board over the player's (or pair's) LAST 128 boards, expressed in
// milli-IMPs (sum / 128 * 1000, so missing boards count as 0 IMP/board).
//
//   - solo bid   : bid-only boards versus the bots            (play_stats.recent)
//   - solo play  : full-play boards versus the bots           (play_stats.recentPlay)
//   - pair bid   : partner-table boards, per host+partner pair (game files)
//
// To be listed a player/pair must have scored at least rankMinRecent boards
// in the last rankRecency (3 months); the ranked window itself is the last
// rankWindow boards regardless of age.

const (
	rankWindow    = 128 // boards in the ranking average
	rankMinRecent = 16  // min boards within rankRecency to be listed
	rankRecency   = 90 * 24 * time.Hour
	rankListMax   = 100 // entries returned per leaderboard
)

type rankEntry struct {
	Name      string `json:"name"`
	Partner   string `json:"partner,omitempty"` // pair leaderboard only
	MilliImps int    `json:"milliImps"`         // avg mIMP/board over the last 128
	Boards    int    `json:"boards"`            // boards inside the 128-window
	Recent    int    `json:"recent"`            // boards in the last 3 months
}

// milliAvg converts a newest-first IMP list (already capped at 128) into the
// milli-IMP average, counting boards short of 128 as 0 IMP.
func milliAvg(imps []float64) int {
	sum := 0.0
	for _, v := range imps {
		sum += v
	}
	return int(math.Round(sum / float64(rankWindow) * 1000))
}

func countRecent(ts []time.Time, cutoff time.Time) int {
	n := 0
	for _, t := range ts {
		if !t.IsZero() && t.After(cutoff) {
			n++
		}
	}
	return n
}

func sortRankEntries(list []rankEntry) []rankEntry {
	sort.Slice(list, func(i, j int) bool {
		a, b := list[i], list[j]
		if a.MilliImps != b.MilliImps {
			return a.MilliImps > b.MilliImps
		}
		if a.Recent != b.Recent {
			return a.Recent > b.Recent
		}
		if a.Name != b.Name {
			return a.Name < b.Name
		}
		return a.Partner < b.Partner
	})
	if len(list) > rankListMax {
		return list[:rankListMax]
	}
	return list
}

func parseTs(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}
	}
	return t
}

// soloRank builds one solo leaderboard from the given per-user record lists.
func soloRank(pick func(PlayStats) []PlayRecord, cutoff time.Time) []rankEntry {
	playMu.Lock()
	snapshot := make(map[string][]PlayRecord, len(playStats))
	for u, s := range playStats {
		recs := pick(s)
		cp := make([]PlayRecord, len(recs))
		copy(cp, recs)
		snapshot[u] = cp
	}
	playMu.Unlock()

	out := make([]rankEntry, 0, len(snapshot))
	for u, recs := range snapshot {
		if len(recs) == 0 {
			continue
		}
		if len(recs) > rankWindow {
			recs = recs[:rankWindow]
		}
		imps := make([]float64, 0, len(recs))
		ts := make([]time.Time, 0, len(recs))
		for _, r := range recs {
			imps = append(imps, r.Imps)
			ts = append(ts, parseTs(r.Ts))
		}
		recent := countRecent(ts, cutoff)
		if recent < rankMinRecent {
			continue
		}
		out = append(out, rankEntry{
			Name: u, MilliImps: milliAvg(imps), Boards: len(recs), Recent: recent,
		})
	}
	return sortRankEntries(out)
}

// pairRank aggregates every scored partner-table board per host+partner pair.
func pairRank(cutoff time.Time) []rankEntry {
	teamMu.Lock()
	games := make([]*TeamGame, 0, len(teamGames))
	for _, g := range teamGames {
		games = append(games, g)
	}
	teamMu.Unlock()

	type board struct {
		ts   time.Time
		imps float64
	}
	type pairKey struct{ a, b string }
	byPair := map[pairKey][]board{}
	for _, g := range games {
		g.mu.Lock()
		if g.Host == "" || g.Partner == "" {
			g.mu.Unlock()
			continue
		}
		k := pairKey{g.Host, g.Partner}
		if k.b < k.a {
			k.a, k.b = k.b, k.a
		}
		for _, b := range g.Boards {
			if b.Result == nil {
				continue
			}
			ts := parseTs(b.Ts)
			if ts.IsZero() {
				ts = g.Created.UTC() // boards scored before timestamps existed
			}
			byPair[k] = append(byPair[k], board{ts: ts, imps: b.Result.Imps})
		}
		g.mu.Unlock()
	}

	out := make([]rankEntry, 0, len(byPair))
	for k, boards := range byPair {
		if len(boards) == 0 {
			continue
		}
		sort.SliceStable(boards, func(i, j int) bool {
			return boards[i].ts.Before(boards[j].ts)
		})
		recent := 0
		for _, b := range boards {
			if b.ts.After(cutoff) {
				recent++
			}
		}
		if recent < rankMinRecent {
			continue
		}
		if len(boards) > rankWindow {
			boards = boards[len(boards)-rankWindow:] // keep the newest 128
		}
		imps := make([]float64, 0, len(boards))
		for _, b := range boards {
			imps = append(imps, b.imps)
		}
		out = append(out, rankEntry{
			Name: k.a, Partner: k.b, MilliImps: milliAvg(imps),
			Boards: len(boards), Recent: recent,
		})
	}
	return sortRankEntries(out)
}

// GET /api/play/rankings - the three leaderboards
func handleRankings(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	cutoff := time.Now().Add(-rankRecency)
	empty := []rankEntry{}
	soloBid := soloRank(func(s PlayStats) []PlayRecord { return s.Recent }, cutoff)
	soloPlay := soloRank(func(s PlayStats) []PlayRecord { return s.RecentPlay }, cutoff)
	pairBid := pairRank(cutoff)
	if soloBid == nil {
		soloBid = empty
	}
	if soloPlay == nil {
		soloPlay = empty
	}
	if pairBid == nil {
		pairBid = empty
	}
	writeJSON(w, map[string]any{
		"soloBid":   soloBid,
		"soloPlay":  soloPlay,
		"pairBid":   pairBid,
		"window":    rankWindow,
		"minRecent": rankMinRecent,
	})
}
