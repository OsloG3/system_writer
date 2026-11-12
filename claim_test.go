package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// ---- Claim verification tests ----
//
// Every scenario is a full 52-card deal plus the list of cards each seat must
// still hold at the claim ("keep"); claimTestPlays legally discards everything
// else, so the plays it returns always lead to the same end position.

// claimHandsA is an NT deal whose four aces and kings (spades and hearts) sit
// with N-S, the queens and jacks with E-W.
var claimHandsA = [4]string{
	"AT3.AT2.843.AJT3", // N
	"Q65.Q96.AJT9.987", // E
	"K94.K85.765.KQ65", // S
	"J872.J743.KQ2.42", // W
}

// claimHandsD is a heart deal for the ruffing test: East holds NO heart (so
// the plays prove it void), and at the claim South holds the top spades while
// North holds only the two top trumps.
var claimHandsD = [4]string{
	"654.AKQ63.QJ9.54", // N
	"JT8..K8762.AJT63", // E (void in hearts)
	"AK32.JT92.A5.KQ9", // S
	"Q97.8754.T43.872", // W
}

// claimTestPlays discards every card outside keep[seat] in legal tricks: the
// leader of trick t is seat t%4, everyone follows the led suit with a
// DISCARDABLE card when they hold the suit and throws a discard otherwise, so
// every keep survives and every void the plays imply is real. The schedule is
// found by depth-first search with backtracking (a plain greedy order can
// deadlock a leader whose only discards left are in suits nobody can follow).
// claimTestPlaysPlanned is claimTestPlays with an explicit per-trick lead
// suit plan ("DDSDS..."), one char per trick, filled deterministically: the
// leader of trick t is (opener+t)%4 and leads its highest discardable card of
// the planned suit; followers follow with their highest discardable card of
// that suit, or dump per dumpOrder when void. It fails the test if any trick
// of the plan is illegal.
func claimTestPlaysPlanned(t *testing.T, hands [4]string, keep [4][]string, opener int, plan string) []PlayCard {
	t.Helper()
	h := newClaimPlayHelper(t, hands, keep, "SHDC", opener)
	for ti := 0; ti < len(plan); ti++ {
		led := suitNum(plan[ti])
		leader := (opener + ti) % 4
		leadCard := ""
		for _, c := range h.disc(leader) {
			if suitNum(cardSuit(c)) == led {
				leadCard = c
				break
			}
		}
		if leadCard == "" {
			t.Fatalf("plan trick %d: seat %d cannot lead %c", ti, leader, plan[ti])
		}
		h.remove(leader, leadCard)
		h.plays = append(h.plays, PlayCard{Seat: leader, Card: leadCard})
		for other := 1; other < 4; other++ {
			f := (leader + other) % 4
			card := ""
			for _, x := range h.disc(f) {
				if suitNum(cardSuit(x)) == led {
					card = x
					break
				}
			}
			if card == "" {
				if ds := h.plannedDump(f, plan, ti); ds != "" {
					card = ds
				}
			}
			if card == "" {
				t.Fatalf("plan trick %d: seat %d has no legal play", ti, f)
			}
			h.remove(f, card)
			h.plays = append(h.plays, PlayCard{Seat: f, Card: card})
		}
	}
	for seat := 0; seat < 4; seat++ {
		if len(h.disc(seat)) != 0 {
			t.Fatalf("plan left discards for seat %d: %v", seat, h.disc(seat))
		}
	}
	return h.plays
}

func newClaimPlayHelper(t *testing.T, hands [4]string, keep [4][]string, suitOrder string, opener int) *claimPlayHelper {
	t.Helper()
	h := &claimPlayHelper{suitOrder: suitOrder, opener: opener}
	for seat := 0; seat < 4; seat++ {
		h.rem[seat] = handCards(hands[seat])
		h.keepSet[seat] = map[string]bool{}
		for _, c := range keep[seat] {
			if !slices.Contains(h.rem[seat], c) {
				t.Fatalf("seat %d does not hold kept card %s", seat, c)
			}
			h.keepSet[seat][c] = true
		}
	}
	n := len(h.disc(0))
	for seat := 1; seat < 4; seat++ {
		if len(h.disc(seat)) != n {
			t.Fatalf("seat %d has %d discards, seat 0 has %d -- every seat must discard equally", seat, len(h.disc(seat)), n)
		}
	}
	return h
}

func claimTestPlays(t *testing.T, hands [4]string, keep [4][]string, suitOrder string, opener int) []PlayCard {
	t.Helper()
	h := newClaimPlayHelper(t, hands, keep, suitOrder, opener)
	if !h.run() {
		t.Fatalf("could not play the discards down legally (hands %v keep %v)", hands, keep)
	}
	return h.plays
}

type claimPlayHelper struct {
	rem       [4][]string
	keepSet   [4]map[string]bool
	suitOrder string
	opener    int
	plays     []PlayCard
	nodes     int
}

func (h *claimPlayHelper) disc(seat int) []string {
	var out []string
	for _, c := range h.rem[seat] {
		if !h.keepSet[seat][c] {
			out = append(out, c)
		}
	}
	return out
}

func (h *claimPlayHelper) remove(seat int, card string) int {
	for i, c := range h.rem[seat] {
		if c == card {
			h.rem[seat] = append(h.rem[seat][:i], h.rem[seat][i+1:]...)
			return i
		}
	}
	panic("card not held: " + card)
}

func (h *claimPlayHelper) restore(seat, idx int, card string) {
	h.rem[seat] = append(h.rem[seat], "")
	copy(h.rem[seat][idx+1:], h.rem[seat][idx:])
	h.rem[seat][idx] = card
}

func (h *claimPlayHelper) run() bool {
	h.nodes++
	if h.nodes > 3_000_000 {
		return false
	}
	total := 0
	for seat := 0; seat < 4; seat++ {
		total += len(h.disc(seat))
	}
	if total == 0 {
		return true
	}
	leader := (h.opener + len(h.plays)/4) % 4
	// candidate leads: prefer suits that still hold the most discards table-wide
	// (they get blocked soonest -- a keep in the suit ends its leads for good),
	// then the caller's suit order, then the hand order
	type cand struct {
		card string
		tot  int
		so   int
	}
	var cands []cand
	for _, c := range h.disc(leader) {
		s := cardSuit(c)
		tot := 0
		for seat := 0; seat < 4; seat++ {
			for _, x := range h.rem[seat] {
				if cardSuit(x) == s && !h.keepSet[seat][x] {
					tot++
				}
			}
		}
		cands = append(cands, cand{c, tot, strings.IndexByte(h.suitOrder, s)})
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].tot != cands[j].tot {
			return cands[i].tot > cands[j].tot
		}
		return cands[i].so < cands[j].so
	})
	for _, cd := range cands {
		led := cardSuit(cd.card)
		var opts [3][]string
		valid := true
		for other := 1; other < 4; other++ {
			f := (leader + other) % 4
			holds := false
			for _, x := range h.rem[f] {
				if cardSuit(x) == led {
					holds = true
					break
				}
			}
			if holds {
				for _, x := range h.disc(f) {
					if cardSuit(x) == led {
						opts[other-1] = append(opts[other-1], x)
					}
				}
			} else {
				opts[other-1] = h.dumpOrder(f)
			}
			if len(opts[other-1]) == 0 {
				valid = false
				break
			}
		}
		if valid && h.tryCombo(leader, cd.card, opts, 0, [3]string{}) {
			return true
		}
	}
	return false
}

// plannedDump picks the off-suit discard of a seat that is void in the led
// suit: the card of the suit whose remaining count exceeds the follow slots
// the rest of the plan still offers it (that surplus has to be dumped
// somewhere, so dump it now); ties go to the hand order.
func (h *claimPlayHelper) plannedDump(seat int, plan string, trick int) string {
	ds := h.disc(seat)
	if len(ds) == 0 {
		return ""
	}
	rest := map[int8]int{}
	for ti := trick + 1; ti < len(plan); ti++ {
		rest[suitNum(plan[ti])]++
	}
	cnt := map[int8]int{}
	for _, c := range ds {
		cnt[suitNum(cardSuit(c))]++
	}
	best, bestCard := -99, ds[0]
	for _, c := range ds {
		s := suitNum(cardSuit(c))
		if sur := cnt[s] - rest[s]; sur > best {
			best, bestCard = sur, c
		}
	}
	return bestCard
}

// dumpOrder ranks the discards of a seat that is void in the led suit: cards
// that empty their suit first (being void unlocks future discards), then the
// longest suit first (short suits are scarcer follow material).
func (h *claimPlayHelper) dumpOrder(seat int) []string {
	ds := h.disc(seat)
	count := map[byte]int{}
	for _, c := range ds {
		count[cardSuit(c)]++
	}
	out := append([]string(nil), ds...)
	sort.SliceStable(out, func(i, j int) bool {
		ci, cj := count[cardSuit(out[i])], count[cardSuit(out[j])]
		if (ci == 1) != (cj == 1) {
			return ci == 1
		}
		return ci > cj
	})
	return out
}

// tryCombo enumerates the three followers' answers (capped per follower to
// keep the search small) and recurses into the next trick.
func (h *claimPlayHelper) tryCombo(leader int, lc string, opts [3][]string, k int, picks [3]string) bool {
	if k < 3 {
		limit := min(len(opts[k]), 8)
		for i := 0; i < limit; i++ {
			picks[k] = opts[k][i]
			if h.tryCombo(leader, lc, opts, k+1, picks) {
				return true
			}
		}
		return false
	}
	idxL := h.remove(leader, lc)
	seats := [3]int{(leader + 1) % 4, (leader + 2) % 4, (leader + 3) % 4}
	idxs := [3]int{}
	for i := 0; i < 3; i++ {
		idxs[i] = h.remove(seats[i], picks[i])
	}
	h.plays = append(h.plays,
		PlayCard{Seat: leader, Card: lc},
		PlayCard{Seat: seats[0], Card: picks[0]},
		PlayCard{Seat: seats[1], Card: picks[1]},
		PlayCard{Seat: seats[2], Card: picks[2]},
	)
	if h.run() {
		return true
	}
	h.plays = h.plays[:len(h.plays)-4]
	for i := 2; i >= 0; i-- {
		h.restore(seats[i], idxs[i], picks[i])
	}
	h.restore(leader, idxL, lc)
	return false
}

func claimSide(decl, dummy int) (claim, known [4]bool) {
	claim[decl], claim[dummy] = true, true
	known[decl], known[dummy] = true, true
	return
}

// The classic "the rest are mine": N-S keep the spade and heart aces and kings
// over the defenders' queens and jacks, so declarer (South) claims the last two
// tricks from dummy's lead -- and the claim holds for BOTH layouts of E/W.
func TestVerifyClaimAccepted(t *testing.T) {
	keep := [4][]string{{"AS", "AH"}, {"QS", "QH"}, {"KS", "KH"}, {"JS", "JH"}}
	plays := claimTestPlays(t, claimHandsA, keep, "SHDC", 0)
	if len(plays) != 44 {
		t.Fatalf("played %d cards, want 44", len(plays))
	}
	claim, known := claimSide(2, 0)
	ok, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 0, 2)
	if err != nil || !ok {
		t.Fatalf("claim refused: ok=%v err=%v", ok, err)
	}
	// a claim by a seat that is not on the claim side is refused outright
	if _, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 1, 2); err == nil {
		t.Fatal("East must not be able to claim for N-S")
	}
}

// The defense keeps a card that beats the claim side's run: the claim must be
// refused (the spade and heart queens take a trick in every layout).
func TestVerifyClaimRefused(t *testing.T) {
	keep := [4][]string{{"3S", "2H"}, {"QS", "QH"}, {"4S", "5H"}, {"JS", "JH"}}
	plays := claimTestPlays(t, claimHandsA, keep, "SHDC", 0)
	claim, known := claimSide(2, 0)
	ok, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 0, 2)
	if ok {
		t.Fatal("a claim the defense can beat was accepted")
	}
	if err != nil && !errors.Is(err, errClaimTimeout) {
		t.Fatalf("unexpected error: %v", err)
	}
}

// A partial claim: North keeps the spade ace and a small heart, South the
// spade king and a club. North wins the FIRST trick with the spade ace in
// every layout, but the heart two on the second trick loses to the defense's
// queen or jack -- so claiming 1 is accepted and claiming 2 refused. A claim
// of 0 (a concede) is auto-accepted, and claiming more than remains is an
// error.
func TestVerifyClaimPartial(t *testing.T) {
	keep := [4][]string{{"AS", "2H"}, {"QS", "QH"}, {"KS", "KC"}, {"JS", "JH"}}
	plays := claimTestPlaysPlanned(t, claimHandsA, keep, 0, "SDHSDCHDCDC")
	if len(plays) != 44 {
		t.Fatalf("played %d cards, want 44", len(plays))
	}
	claim, known := claimSide(2, 0)
	if ok, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 0, 1); err != nil || !ok {
		t.Fatalf("one-trick claim refused: ok=%v err=%v", ok, err)
	}
	if ok, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 0, 2); ok || err != nil {
		t.Fatalf("two-trick claim must be refused: ok=%v err=%v", ok, err)
	}
	if ok, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 0, 0); err != nil || !ok {
		t.Fatalf("a concede must be auto-accepted: ok=%v err=%v", ok, err)
	}
	if _, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 0, 3); err == nil {
		t.Fatal("claiming more tricks than remain must fail")
	}
}

// A claim made in the middle of a trick: West led the jack of spades, North
// covered with the ace and East with the queen; South -- last to play, holding
// just the king of spades and king of hearts -- claims before playing to the
// trick. The claim side wins the current trick and the last one.
func TestVerifyClaimMidTrick(t *testing.T) {
	keep := [4][]string{{"AS", "AH"}, {"QS", "QH"}, {"KS", "KH"}, {"JS", "JH"}}
	plays := claimTestPlays(t, claimHandsA, keep, "SHDC", 0)
	if len(plays) != 44 {
		t.Fatalf("played %d cards, want 44", len(plays))
	}
	// trick 12: West leads its jack of spades, North and East play their
	// spade keeps; South is to act and claims
	plays = append(plays,
		PlayCard{Seat: 3, Card: "JS"},
		PlayCard{Seat: 0, Card: "AS"},
		PlayCard{Seat: 1, Card: "QS"},
	)
	claim, known := claimSide(2, 0)
	ok, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 2, 2)
	if err != nil || !ok {
		t.Fatalf("mid-trick claim refused: ok=%v err=%v", ok, err)
	}
}

// Trumps: South (declarer) runs the top spades while North holds nothing but
// the two top trumps. Whoever ruffs, North overruffs -- but East's proven
// heart void matters: without it East could hold the 5H and... North's trump
// discard still beats it, yet the void must be derived from the plays anyway.
func TestVerifyClaimWithTrumps(t *testing.T) {
	keep := [4][]string{{"AH", "KH"}, {"JS", "2D"}, {"AS", "KS"}, {"9S", "5H"}}
	// hearts first: East holds none, so the very first heart lead proves the
	// void the claim verification depends on
	plays := claimTestPlays(t, claimHandsD, keep, "HSDC", 0)
	if len(plays) != 44 {
		t.Fatalf("played %d cards, want 44", len(plays))
	}
	claim, known := claimSide(2, 0)
	sv, err := buildClaimSolver(claimHandsD, plays, 'H', claim, known, 2)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if sv.voids[1]&(1<<suitNum('H')) == 0 {
		t.Fatal("East's heart void was not derived from the plays")
	}
	ok, err := verifyClaim(claimHandsD, plays, 'H', claim, known, 2, 2)
	if err != nil || !ok {
		t.Fatalf("trump claim refused: ok=%v err=%v", ok, err)
	}
}

// A defender's claim: West keeps the diamond king (the ace fell early) and
// claims the last trick personally -- the tabled dummy is public information
// and plays adversarially, the other two hands are unseen.
func TestVerifyClaimDefender(t *testing.T) {
	keep := [4][]string{{"TH"}, {"6S"}, {"5H"}, {"KD"}}
	// an explicit lead-suit plan: diamonds x2, a spade, the last two diamonds
	// (East leads the fourth -- everyone else is void and dumps), the second
	// spade, a heart, three clubs, a heart, and North's last club
	plays := claimTestPlaysPlanned(t, claimHandsA, keep, 1, "DDSDDSHCCCHC")
	if len(plays) != 48 {
		t.Fatalf("played %d cards, want 48", len(plays))
	}
	var claim, known [4]bool
	claim[3], known[3], known[0] = true, true, true // West claims; dummy is tabled
	ok, err := verifyClaim(claimHandsA, plays, 'N', claim, known, 3, 1)
	if err != nil || !ok {
		t.Fatalf("defender claim refused: ok=%v err=%v", ok, err)
	}
}

// Before the opening lead a lone defender's claim can never be proven (three
// hidden hands, 39 unseen cards): it must be refused, not accepted.
func TestVerifyClaimBeforeLeadRefused(t *testing.T) {
	var claim, known [4]bool
	claim[2], known[2] = true, true // South claims before leading; dummy is face down
	ok, _ := verifyClaim(claimHandsA, nil, 'N', claim, known, 2, 13)
	if ok {
		t.Fatal("a 13-trick defender claim was accepted before the opening lead")
	}
}

// The fast path settles the classic "spread" claim instantly: a declarer hand
// solid in every suit takes all thirteen tricks before a card is played, even
// with all 26 defense cards unseen.
func TestVerifyClaimSolidSpread(t *testing.T) {
	hands := [4]string{
		"5432.543.543.543", // N (dummy)
		"JT9.JT9.JT9.T976", // E
		"AKQ.AKQ.AKQ.AKQJ", // S (declarer)
		"876.8762.8762.82", // W
	}
	claim, known := claimSide(2, 0)
	start := time.Now()
	ok, err := verifyClaim(hands, nil, 'N', claim, known, 2, 13)
	if err != nil || !ok {
		t.Fatalf("solid spread refused: ok=%v err=%v", ok, err)
	}
	if d := time.Since(start); d > 200*time.Millisecond {
		t.Fatalf("solid spread took %v, want instant", d)
	}
}

// A big claim that cannot be proven is refused within the verification budget
// instead of hanging the request.
func TestVerifyClaimBudget(t *testing.T) {
	claim, known := claimSide(2, 0)
	start := time.Now()
	ok, err := verifyClaim(claimHandsA, nil, 'N', claim, known, 2, 13)
	if ok {
		t.Fatal("claiming all 13 tricks on this deal must be refused")
	}
	if err != nil && !errors.Is(err, errClaimTimeout) {
		t.Fatalf("unexpected error: %v", err)
	}
	if d := time.Since(start); d > claimVerifyBudget+2*time.Second {
		t.Fatalf("verification took %v, want within budget", d)
	}
}

func TestClaimRemainingTricks(t *testing.T) {
	want := map[int]int{0: 13, 4: 12, 44: 2, 45: 2, 47: 2, 48: 1, 51: 1}
	for plays, w := range want {
		if got := claimRemainingTricks(plays); got != w {
			t.Fatalf("claimRemainingTricks(%d) = %d, want %d", plays, got, w)
		}
	}
}

// The board- and session-level entry points build the same position: the
// declarer sees both claim-side hands, a defender sees only its own plus the
// tabled dummy.
func TestCheckClaimWiring(t *testing.T) {
	plays := claimTestPlays(t, claimHandsA, claimKeepGood, "SHDC", 0)

	b := &PlayBoard{ID: "claim1", User: "u", Mode: "play", Hands: claimHandsA}
	b.Play = &PlayState{
		Contract: &PlayContract{Level: 3, Denom: "N", Declarer: 2},
		Trump:    'N', Declarer: 2, Dummy: 0, Opener: 3,
		Plays: plays,
	}
	if ok, err := b.checkClaim(2, 2); err != nil || !ok {
		t.Fatalf("board declarer claim refused: ok=%v err=%v", ok, err)
	}

	s := &DrillSession{
		ID: "claim1", User: "u",
		Problem:  DrillProblem{Hands: claimHandsA, Role: "declarer", HeroSeat: 2},
		contract: &PlayContract{Level: 3, Denom: "N", Declarer: 2},
		trump:    'N',
	}
	for _, pc := range plays {
		s.Plays = append(s.Plays, DrillPlay{Seat: pc.Seat, Card: pc.Card})
	}
	if ok, err := s.checkClaim(0, 2); err != nil || !ok { // dummy's turn to lead
		t.Fatalf("drill declarer claim refused: ok=%v err=%v", ok, err)
	}
}

// claimTestBoard builds a play-mode board mid-hand (declarer South, the human
// seat) whose 44 plays leave each seat the two keep cards; DeclTricks matches
// the played tricks.
func claimTestBoard(t *testing.T, id string, keep [4][]string) *PlayBoard {
	t.Helper()
	plays := claimTestPlays(t, claimHandsA, keep, "SHDC", 0)
	b := &PlayBoard{ID: id, User: "claimer", Mode: "play", Hands: claimHandsA, Created: time.Now()}
	b.Play = &PlayState{
		Contract: &PlayContract{Level: 3, Denom: "N", Declarer: 2},
		Trump:    'N', Declarer: 2, Dummy: 0, Opener: 3, Plays: plays,
	}
	for tr := 0; tr*4+4 <= len(plays); tr++ {
		if w := trickWinner(plays[tr*4:tr*4+4], 'N'); w == 0 || w == 2 {
			b.Play.DeclTricks++
		}
	}
	if s := b.Play.nextSeat(); s != 2 {
		t.Fatalf("test position must leave the human on lead, got seat %d", s)
	}
	return b
}

var (
	claimKeepGood = [4][]string{{"AS", "AH"}, {"QS", "QH"}, {"KS", "KH"}, {"JS", "JH"}}
	claimKeepBad  = [4][]string{{"3S", "2H"}, {"QS", "QH"}, {"4S", "5H"}, {"JS", "JH"}}
)

func TestHandlePlayClaim(t *testing.T) {
	oldDir, oldBoards, oldStats, oldURL := dataDir, playBoards, playStats, botURL
	dataDir = t.TempDir()
	if err := os.MkdirAll(gameDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	playBoards = map[string]*PlayBoard{}
	playStats = map[string]PlayStats{}
	t.Cleanup(func() { dataDir, playBoards, playStats, botURL = oldDir, oldBoards, oldStats, oldURL })

	var scored atomic.Int32
	scored.Store(-1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/play/result" {
			http.Error(w, "unexpected "+r.URL.Path, 404)
			return
		}
		var body struct {
			Tricks int `json:"tricks"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		scored.Store(int32(body.Tricks))
		writeJSON(w, map[string]any{
			"contract":  map[string]any{"level": 3, "denom": "N", "declarer": 2, "penalty": 0},
			"dd_tricks": 9, "par_ns": 400, "score_ns": 400, "imps": 0.0,
		})
	}))
	defer srv.Close()
	botURL = srv.URL
	token := drillLogin(t, "claimer")

	// an accepted claim: the human declared, N-S keep the spade and heart
	// bosses, so the last two tricks cannot escape
	good := claimTestBoard(t, "claimgood001", claimKeepGood)
	before := good.Play.DeclTricks
	playBoards[good.ID] = good
	rr := httptest.NewRecorder()
	req := drillReq(t, "POST", "/api/play/"+good.ID+"/claim", `{"tricks":2}`, token)
	req.SetPathValue("id", good.ID)
	handlePlayClaim(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("claim: %d %s", rr.Code, rr.Body.String())
	}
	var st map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &st); err != nil {
		t.Fatal(err)
	}
	if st["phase"] != "done" {
		t.Fatalf("phase = %v, want done (%s)", st["phase"], rr.Body.String())
	}
	if got := scored.Load(); int(got) != before+2 {
		t.Fatalf("scored with %d tricks, want %d", got, before+2)
	}
	play, _ := st["play"].(map[string]any)
	if play == nil || play["claimed"] != true || int(play["declTricks"].(float64)) != before+2 {
		t.Fatalf("bad claimed play state: %v", play)
	}
	res, _ := st["result"].(map[string]any)
	if res == nil || res["played"] == nil || int(res["played"].(float64)) != before+2 {
		t.Fatalf("bad result: %v", res)
	}

	// a refused claim: the defense keeps the boss cards; the board must stay
	// exactly as it was and the answer is a 409
	bad := claimTestBoard(t, "claimbad0001", claimKeepBad)
	playBoards[bad.ID] = bad
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/play/"+bad.ID+"/claim", `{"tricks":2}`, token)
	req.SetPathValue("id", bad.ID)
	handlePlayClaim(rr, req)
	if rr.Code != http.StatusConflict {
		t.Fatalf("refused claim: %d %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "claim rejected") {
		t.Fatalf("bad refusal body: %s", rr.Body.String())
	}
	if bad.Result != nil || bad.Play.Done || bad.Play.Claimed {
		t.Fatal("a refused claim must not touch the board")
	}

	// claiming more tricks than remain is a 400
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/play/"+bad.ID+"/claim", `{"tricks":3}`, token)
	req.SetPathValue("id", bad.ID)
	handlePlayClaim(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("over-claim: %d %s", rr.Code, rr.Body.String())
	}

	// a concede (0 tricks) is auto-accepted even on this hopeless position:
	// the declarer side simply takes no more tricks
	badBefore := bad.Play.DeclTricks
	scored.Store(-1)
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/play/"+bad.ID+"/claim", `{"tricks":0}`, token)
	req.SetPathValue("id", bad.ID)
	handlePlayClaim(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("concede: %d %s", rr.Code, rr.Body.String())
	}
	if got := scored.Load(); int(got) != badBefore {
		t.Fatalf("concede scored with %d tricks, want %d", got, badBefore)
	}
	if bad.Result == nil || !bad.Play.Done || !bad.Play.Claimed || bad.Play.DeclTricks != badBefore {
		t.Fatalf("bad concede result: %+v", bad.Play)
	}

	// logged out
	rr = httptest.NewRecorder()
	handlePlayClaim(rr, drillReq(t, "POST", "/api/play/"+good.ID+"/claim", "", ""))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("logged-out claim: %d", rr.Code)
	}
}

func TestHandleDrillSessionClaim(t *testing.T) {
	oldSess := drillSess
	drillSess = map[string]*DrillSession{}
	t.Cleanup(func() { drillSess = oldSess })
	token := drillLogin(t, "hero")

	claimSession := func(id string, plays []PlayCard) *DrillSession {
		s := &DrillSession{
			ID: id, User: "hero", Created: time.Now(),
			Problem: DrillProblem{
				ID: "prob1", Hands: claimHandsA, Role: "declarer", HeroSeat: 2,
				Difficulty: 3, Target: 8, Lines: []DrillLine{{}},
			},
			contract: &PlayContract{Level: 3, Denom: "N", Declarer: 2},
			trump:    'N',
			Live:     []int{0},
		}
		for _, pc := range plays {
			s.applyLocked(DrillPlay{Seat: pc.Seat, Card: pc.Card})
		}
		drillSess[id] = s
		return s
	}

	// accepted: the declarer hero claims the last two tricks on their lead
	good := claimSession("drillgood001", claimTestPlays(t, claimHandsA, claimKeepGood, "SHDC", 0))
	before := good.DeclTricks
	rr := httptest.NewRecorder()
	req := drillReq(t, "POST", "/api/drill/session/"+good.ID+"/claim", `{"tricks":2}`, token)
	req.SetPathValue("sid", good.ID)
	handleDrillSessionClaim(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("drill claim: %d %s", rr.Code, rr.Body.String())
	}
	var st map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &st); err != nil {
		t.Fatal(err)
	}
	if st["phase"] != "done" {
		t.Fatalf("phase = %v, want done", st["phase"])
	}
	res, _ := st["result"].(map[string]any)
	if res == nil || res["claimed"] != true || int(res["tricks"].(float64)) != before+2 || res["success"] != true {
		t.Fatalf("bad drill result: %v", res)
	}

	// refused: a claim the defense can beat leaves the session running
	bad := claimSession("drillbad0001", claimTestPlays(t, claimHandsA, claimKeepBad, "SHDC", 0))
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/drill/session/"+bad.ID+"/claim", `{"tricks":2}`, token)
	req.SetPathValue("sid", bad.ID)
	handleDrillSessionClaim(rr, req)
	if rr.Code != http.StatusConflict {
		t.Fatalf("refused drill claim: %d %s", rr.Code, rr.Body.String())
	}
	if bad.Done || bad.Result != nil {
		t.Fatal("a refused claim must not end the session")
	}
	// a concede (0 tricks) is auto-accepted: the run ends with the tricks
	// declarer already had, graded (here: below target, so it fails)
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/drill/session/"+bad.ID+"/claim", `{"tricks":0}`, token)
	req.SetPathValue("sid", bad.ID)
	handleDrillSessionClaim(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("drill concede: %d %s", rr.Code, rr.Body.String())
	}
	var cst map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &cst); err != nil {
		t.Fatal(err)
	}
	cres, _ := cst["result"].(map[string]any)
	if cres == nil || cres["claimed"] != true || int(cres["tricks"].(float64)) != bad.DeclTricks {
		t.Fatalf("bad concede result: %v", cres)
	}

	// out of turn: a fresh session waits on the opener (West), not the hero
	fresh := claimSession("drillfresh001", nil)
	if fresh.nextSeatLocked() == 2 {
		t.Fatal("a fresh session should wait on the opener")
	}
	rr = httptest.NewRecorder()
	req = drillReq(t, "POST", "/api/drill/session/"+fresh.ID+"/claim", `{"tricks":1}`, token)
	req.SetPathValue("sid", fresh.ID)
	handleDrillSessionClaim(rr, req)
	if rr.Code != http.StatusConflict || !strings.Contains(rr.Body.String(), "your turn") {
		t.Fatalf("out-of-turn claim: %d %s", rr.Code, rr.Body.String())
	}
}
