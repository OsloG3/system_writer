package main

import (
	"slices"
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
