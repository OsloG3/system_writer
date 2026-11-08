package main

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// The partner tables share the site auction cap and the bot auto-pass: past
// the model's MAX_CALLS the East/West bots must pass without the sidecar
// being asked for a bid it would refuse.
func TestTeamAdvanceBotsPassPastModelCap(t *testing.T) {
	var bids int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/bid" {
			atomic.AddInt32(&bids, 1)
		}
		http.Error(w, "the bots must not be asked past the model cap", 500)
	}))
	defer srv.Close()
	oldURL := botURL
	botURL = srv.URL
	defer func() { botURL = oldURL }()

	// 37 calls (the last a bid, so the auction still runs): East is next and
	// the count is already past the model cap
	calls := make([]string, 0, botMaxModelCalls+1)
	for i := 0; i < botMaxModelCalls; i++ {
		calls = append(calls, "P")
	}
	calls = append(calls, "1C")
	g := &TeamGame{ID: "teamtest0000001", Host: "alice", Partner: "bob"}
	b := &TeamBoard{Dealer: 0, Vuln: 0,
		Hands: [4]string{"AT7.KT943.A42.K8", "J6543.75.K8.AQJT", "K982.AQ6.T73.952", "Q.J82.QJ965.7643"},
		Calls: calls}

	if err := g.advanceLocked(b); err != nil {
		t.Fatalf("advanceLocked: %v", err)
	}
	// East passed automatically; South (a human) is next, so no scoring yet
	if len(b.Calls) != botMaxModelCalls+2 || b.Calls[botMaxModelCalls+1] != "P" {
		t.Fatalf("bot did not auto-pass: %v", b.Calls[botMaxModelCalls:])
	}
	if atomic.LoadInt32(&bids) != 0 {
		t.Fatal("the sidecar was asked for a bid past the model cap")
	}
	if b.Result != nil {
		t.Fatal("the board must not be scored while a human still has to act")
	}
	if next := (b.Dealer + len(b.Calls)) % 4; next != teamSouthSeat {
		t.Fatalf("next seat = %d, want South (the host)", next)
	}
}
