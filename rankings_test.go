package main

import (
	"testing"
	"time"
)

func TestMilliAvg(t *testing.T) {
	// 20 boards of +1 IMP: 20/128*1000 = 156.25 -> 156
	imps := make([]float64, 20)
	for i := range imps {
		imps[i] = 1
	}
	if got := milliAvg(imps); got != 156 {
		t.Fatalf("milliAvg = %d, want 156", got)
	}
	// a full 128-board window of +2 IMPs/board is exactly 2000
	full := make([]float64, 128)
	for i := range full {
		full[i] = 2
	}
	if got := milliAvg(full); got != 2000 {
		t.Fatalf("milliAvg = %d, want 2000", got)
	}
	if got := milliAvg(nil); got != 0 {
		t.Fatalf("milliAvg(nil) = %d, want 0", got)
	}
	// negatives: 128 boards of -0.5 -> -500
	neg := make([]float64, 128)
	for i := range neg {
		neg[i] = -0.5
	}
	if got := milliAvg(neg); got != -500 {
		t.Fatalf("milliAvg = %d, want -500", got)
	}
}

func recs(n int, imps float64, ts string) []PlayRecord {
	out := make([]PlayRecord, n)
	for i := range out {
		out[i] = PlayRecord{Ts: ts, Imps: imps}
	}
	return out
}

func TestSoloRankGatesAndOrder(t *testing.T) {
	old := playStats
	defer func() { playStats = old }()
	now := time.Now().UTC().Format(time.RFC3339)
	oldTs := time.Now().Add(-100 * 24 * time.Hour).UTC().Format(time.RFC3339)
	playStats = map[string]PlayStats{
		"alice": {Recent: recs(20, 1, now)},     // eligible: 156 mIMP
		"bob":   {Recent: recs(10, 5, now)},     // too few recent boards
		"carol": {Recent: recs(20, 5, oldTs)},   // all outside 3 months
		"dave":  {Recent: recs(128, 2, now)},    // eligible: 2000 mIMP
		"erin":  {RecentPlay: recs(30, 1, now)}, // play leaderboard: 234
		"frank": {Recent: recs(300, 1, now)},    // capped to the last 128
	}
	cutoff := time.Now().Add(-rankRecency)

	bid := soloRank(func(s PlayStats) []PlayRecord { return s.Recent }, cutoff)
	// order: dave 2000, frank 1000 (128*1/128*1000), alice 156
	if len(bid) != 3 {
		t.Fatalf("want 3 eligible, got %d: %+v", len(bid), bid)
	}
	if bid[0].Name != "dave" || bid[0].MilliImps != 2000 {
		t.Fatalf("top entry wrong: %+v", bid[0])
	}
	if bid[1].Name != "frank" || bid[1].MilliImps != 1000 || bid[1].Boards != 128 {
		t.Fatalf("second entry wrong: %+v", bid[1])
	}
	if bid[2].Name != "alice" || bid[2].MilliImps != 156 {
		t.Fatalf("third entry wrong: %+v", bid[2])
	}

	play := soloRank(func(s PlayStats) []PlayRecord { return s.RecentPlay }, cutoff)
	if len(play) != 1 || play[0].Name != "erin" || play[0].MilliImps != 234 {
		t.Fatalf("play leaderboard wrong: %+v", play)
	}
}

func TestPairRankAggregatesTables(t *testing.T) {
	oldGames := teamGames
	defer func() { teamGames = oldGames }()
	now := time.Now().UTC().Format(time.RFC3339)

	scored := func(n int, imps float64) []*TeamBoard {
		out := make([]*TeamBoard, n)
		for i := range out {
			out[i] = &TeamBoard{Ts: now, Result: &PlayResult{Imps: imps}}
		}
		return out
	}
	g1 := &TeamGame{ID: "g1", Host: "x", Partner: "y", Boards: scored(20, 1), Created: time.Now()}
	g2 := &TeamGame{ID: "g2", Host: "y", Partner: "x", Boards: scored(4, 1), Created: time.Now()} // same pair, other table
	g3 := &TeamGame{ID: "g3", Host: "p", Partner: "q", Boards: scored(5, 9), Created: time.Now()} // too few
	g4 := &TeamGame{ID: "g4", Host: "m", Partner: "n",
		Boards:  []*TeamBoard{{Result: &PlayResult{Imps: 3}}}, // no Ts: falls back to Created
		Created: time.Now().Add(-24 * time.Hour)}
	teamGames = map[string]*TeamGame{"g1": g1, "g2": g2, "g3": g3, "g4": g4}

	out := pairRank(time.Now().Add(-rankRecency))
	if len(out) != 1 {
		t.Fatalf("want 1 eligible pair, got %d: %+v", len(out), out)
	}
	// x+y merged across both tables: 24 boards of +1 -> 24/128*1000 = 187.5 -> 188
	top := out[0]
	if top.Name != "x" || top.Partner != "y" || top.Boards != 24 || top.MilliImps != 188 {
		t.Fatalf("merged pair wrong: %+v", top)
	}
}
