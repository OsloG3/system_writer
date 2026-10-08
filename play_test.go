package main

import (
	"slices"
	"testing"
)

func TestHandCardsAndNormalizeCard(t *testing.T) {
	got := handCards("AT7.KT943.A42.K8")
	want := []string{"AS", "TS", "7S", "KH", "TH", "9H", "4H", "3H",
		"AD", "4D", "2D", "KC", "8C"}
	if !slices.Equal(got, want) {
		t.Fatalf("handCards = %v, want %v", got, want)
	}
	if normalizeCard("as") != "AS" || normalizeCard(" Td ") != "TD" {
		t.Fatalf("normalizeCard rank+suit failed")
	}
	if normalizeCard("SA") != "AS" { // suit+rank tolerated
		t.Fatalf("normalizeCard suit+rank failed")
	}
	if normalizeCard("1S") != "" || normalizeCard("AX") != "" {
		t.Fatalf("normalizeCard accepted junk")
	}
	if cardSuit("9H") != 'H' || cardRankIdx("AS") != 0 || cardRankIdx("2C") != 12 {
		t.Fatalf("card accessors wrong")
	}
}

func TestTrickWinner(t *testing.T) {
	// N leads S5, E plays S9, S plays SA, W plays 2C: South (2) wins.
	trick := []PlayCard{{0, "5S"}, {1, "9S"}, {2, "AS"}, {3, "2C"}}
	if w := trickWinner(trick, 'H'); w != 2 {
		t.Fatalf("winner = %d, want 2", w)
	}
	// Same trick, but W trumps with the 2H: West (3) wins.
	trick[3] = PlayCard{3, "2H"}
	if w := trickWinner(trick, 'H'); w != 3 {
		t.Fatalf("trumped winner = %d, want 3", w)
	}
	// NT: the trump char never matches, so South still wins.
	trick[3] = PlayCard{3, "2C"}
	if w := trickWinner(trick, 'N'); w != 2 {
		t.Fatalf("NT winner = %d, want 2", w)
	}
}

func TestContractFromCalls(t *testing.T) {
	// N opens 1H, E passes, S raises to 4H: declarer is North (first of the
	// N-S side to name hearts).
	c := contractFromCalls(0, []string{"1H", "P", "4H", "P", "P", "P"})
	if c == nil || c.Level != 4 || c.Denom != "H" || c.Declarer != 0 || c.Penalty != 0 {
		t.Fatalf("4H by N wrong: %+v", c)
	}
	// The transfer case: N bids 1H then S names hearts first? Here E opens 1S,
	// W raises to 4S -> declarer is East (first E-W to name spades).
	c = contractFromCalls(1, []string{"1S", "P", "4S", "P", "P", "P"})
	if c == nil || c.Declarer != 1 || c.Denom != "S" {
		t.Fatalf("4S by E wrong: %+v", c)
	}
	// Doubled: 1C by S, X by W, P P -> penalty 1, declarer South.
	c = contractFromCalls(2, []string{"1C", "X", "P", "P", "P"})
	if c == nil || c.Declarer != 2 || c.Penalty != 1 {
		t.Fatalf("1CX by S wrong: %+v", c)
	}
	// Passed out.
	if c := contractFromCalls(0, []string{"P", "P", "P", "P"}); c != nil {
		t.Fatalf("pass-out should be nil, got %+v", c)
	}
}

// playBoard builds a board mid-play with a fixed deal for state-machine tests.
func playBoardTest(t *testing.T, declarer int) *PlayBoard {
	t.Helper()
	hands := [4]string{
		"AT7.KT943.A42.K8", "J6543.75.K8.AQJT",
		"K982.AQ6.T73.952", "Q.J82.QJ965.7643",
	}
	b := &PlayBoard{ID: "b1", User: "u", Hands: hands, Dealer: 0, Vuln: 0}
	b.Play = &PlayState{
		Contract: &PlayContract{Level: 4, Denom: "H", Declarer: declarer},
		Trump:    'H', Declarer: declarer,
		Dummy: (declarer + 2) % 4, Opener: (declarer + 1) % 4,
	}
	return b
}

func TestPlayNextSeatAndLegal(t *testing.T) {
	b := playBoardTest(t, 0) // North declares; opener is East
	if s := b.Play.nextSeat(); s != 1 {
		t.Fatalf("opener = %d, want 1", s)
	}
	// East leads: any of East's cards is legal.
	if len(b.legalCards(1)) != 13 {
		t.Fatalf("opening lead should allow all 13 cards")
	}
	// East leads a spade; South must follow spades (holds K982).
	b.applyPlay(1, "3S")
	if s := b.Play.nextSeat(); s != 2 {
		t.Fatalf("next = %d, want 2", s)
	}
	legal := b.legalCards(2)
	if len(legal) != 4 || !slices.Contains(legal, "KS") {
		t.Fatalf("South must follow spades, got %v", legal)
	}
	if err := b.applyPlay(2, "KH"); err == nil {
		t.Fatalf("expected a revoke to be rejected")
	}
	b.applyPlay(2, "KS")
	b.applyPlay(3, "QS") // West holds only the Q of spades
	b.applyPlay(0, "AS") // North wins with the ace
	if got := b.Play.DeclTricks; got != 1 {
		t.Fatalf("declarer tricks = %d, want 1", got)
	}
	// North won the trick, so North leads again.
	if s := b.Play.nextSeat(); s != 0 {
		t.Fatalf("next leader = %d, want 0", s)
	}
	if got := len(b.remainingHand(0)); got != 12 {
		t.Fatalf("North should hold 12 cards, got %d", got)
	}
}

func TestHumanControlsPlay(t *testing.T) {
	// Human is South (2).
	b := playBoardTest(t, 2) // human declares -> plays South and dummy (North)
	if !b.humanControlsPlay(2) || !b.humanControlsPlay(0) {
		t.Fatalf("human declarer should control both hands")
	}
	if b.humanControlsPlay(1) || b.humanControlsPlay(3) {
		t.Fatalf("bots control the defenders")
	}
	if got := b.humanPlaySeats(); !slices.Equal(got, []int{2, 0}) {
		t.Fatalf("humanPlaySeats = %v", got)
	}

	b = playBoardTest(t, 0) // North declares -> human South is the dummy
	if b.humanControlsPlay(2) {
		t.Fatalf("human dummy plays nothing; the declarer bot tables dummy")
	}
	if got := b.humanPlaySeats(); len(got) != 0 {
		t.Fatalf("humanPlaySeats = %v, want empty", got)
	}

	b = playBoardTest(t, 1) // East declares -> human South defends
	if !b.humanControlsPlay(2) || b.humanControlsPlay(0) {
		t.Fatalf("human defender controls only South")
	}
}
