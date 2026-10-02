package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestWaitingLockedAndNewTurns(t *testing.T) {
	g := &TeamGame{ID: "g1", Host: "host", Partner: "part", Boards: []*TeamBoard{
		{Dealer: 0, Calls: nil},                                  // next N (partner) -> board 1
		{Dealer: 1, Calls: []string{"P"}},                        // next S (host) -> board 2
		{Dealer: 2, Calls: []string{"P"}},                        // next W (bot)
		{Dealer: 3, Calls: []string{"P", "P", "P", "P"}},         // next W (bot), passed out
		{Dealer: 0, Calls: []string{"P"}, Result: &PlayResult{}}, // finished
	}}
	w := g.waitingLocked()
	if !slices.Equal(w[teamNorthSeat], []int{1}) || !slices.Equal(w[teamSouthSeat], []int{2}) {
		t.Fatalf("waitingLocked wrong: %v", w)
	}
	// partner already waiting on board 1; host newly waiting on board 2
	fresh := newTurns(map[int][]int{teamNorthSeat: {1}}, w)
	if len(fresh) != 1 || !slices.Equal(fresh[teamSouthSeat], []int{2}) {
		t.Fatalf("newTurns wrong: %v", fresh)
	}
	// no change -> nothing new
	if n := newTurns(w, w); len(n) != 0 {
		t.Fatalf("expected no new turns, got %v", n)
	}
	// a fresh match (empty before) reports everything
	all := newTurns(nil, w)
	if !slices.Equal(all[teamNorthSeat], []int{1}) || !slices.Equal(all[teamSouthSeat], []int{2}) {
		t.Fatalf("fresh-match newTurns wrong: %v", all)
	}
	// pushNewTurns must not panic when push is not initialized
	g.pushNewTurns(nil, w)
}

func TestSendTurnPushNoVAPID(t *testing.T) {
	// no keys loaded in tests -> silent no-op
	sendTurnPush("someone", "game", []int{3, 4})
}

// Server stores in data/ must never resolve through the public tree API or
// pollute the systems list.
func TestValidIDRejectsSystemFiles(t *testing.T) {
	for _, bad := range []string{"users", "sessions", "play_stats", "vapid", "push_subs", "game_c82ab4485111252e", "game_"} {
		if validID(bad) {
			t.Errorf("validID(%q) = true, want false", bad)
		}
	}
	for _, good := range []string{"0e4ef583162bf20e", "abc-123", generateID()} {
		if !validID(good) {
			t.Errorf("validID(%q) = false, want true", good)
		}
	}
}

func TestMigrateDataLayout(t *testing.T) {
	old := dataDir
	dataDir = t.TempDir()
	defer func() { dataDir = old }()

	write := func(rel, content string) {
		if err := os.WriteFile(filepath.Join(dataDir, rel), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("users.json", `[]`)
	write("sessions.json", `{}`)
	write("vapid.json", `{}`)
	write("push_subs.json", `[]`)
	write("play_stats.json", `{"u":{"boards":1}}`)
	write("0e4ef583162bf20e.json", `{"name":"tree"}`)
	write("game_c82ab4485111252e.json", `{"id":"c82ab4485111252e"}`)

	if err := migrateDataLayout(); err != nil {
		t.Fatal(err)
	}
	exists := func(rel string) bool {
		_, err := os.Stat(filepath.Join(dataDir, rel))
		return err == nil
	}
	for _, p := range []string{"sistems/0e4ef583162bf20e.json", "game/c82ab4485111252e.json", "game/play_stats.json"} {
		if !exists(p) {
			t.Errorf("missing %s after migration", p)
		}
	}
	for _, p := range []string{"users.json", "sessions.json", "vapid.json", "push_subs.json"} {
		if !exists(p) {
			t.Errorf("%s should stay at the data root", p)
		}
	}
	for _, p := range []string{"0e4ef583162bf20e.json", "game_c82ab4485111252e.json", "play_stats.json"} {
		if exists(p) {
			t.Errorf("%s should have moved out of the data root", p)
		}
	}
	// the loaders follow the files
	if tr, err := loadTree("0e4ef583162bf20e"); err != nil || tr.Name != "tree" {
		t.Errorf("loadTree after migration: %v %+v", err, tr)
	}
	if teamGamePath("c82ab4485111252e") != filepath.Join(dataDir, "game", "c82ab4485111252e.json") {
		t.Error("teamGamePath should point into game/")
	}
	if playStatsPath() != filepath.Join(dataDir, "game", "play_stats.json") {
		t.Error("playStatsPath should point into game/")
	}
	// idempotent, and a stale root-level copy never clobbers the new layout
	if err := migrateDataLayout(); err != nil {
		t.Fatal(err)
	}
	write("0e4ef583162bf20e.json", `{"name":"stale"}`)
	if err := migrateDataLayout(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dataDir, "sistems", "0e4ef583162bf20e.json"))
	if err != nil || !strings.Contains(string(data), `"tree"`) {
		t.Errorf("stale root file clobbered the migrated tree: %v %s", err, data)
	}
}
