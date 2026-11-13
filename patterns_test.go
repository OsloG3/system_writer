package main

import (
	"strings"
	"testing"
)

// ---- helpers mirroring tests/test_patterns.py ----

func mustEntries(t *testing.T, lines ...string) []*patEntry {
	t.Helper()
	out := make([]*patEntry, 0, len(lines))
	for _, l := range lines {
		e, err := parseEntryLine(l)
		if err != nil {
			t.Fatalf("parseEntryLine(%q): %v", l, err)
		}
		out = append(out, e)
	}
	return out
}

func callIDs(t *testing.T, calls []string) []int {
	t.Helper()
	out := make([]int, 0, len(calls))
	for _, c := range calls {
		id, ok := vocabID(c)
		if !ok {
			t.Fatalf("bad call %q", c)
		}
		out = append(out, int(id))
	}
	return out
}

// matchAt folds the prefix through the pattern state (like the live walk in
// callAlerts does) and matches the candidate call.
func matchAt(t *testing.T, entries []*patEntry, dealer int, calls []string,
	call string) (*patEntry, map[byte]int) {
	t.Helper()
	st := newMatchState(dealer)
	for i, c := range calls {
		id, ok := vocabID(c)
		if !ok {
			t.Fatalf("bad call %q", c)
		}
		foldState(entries, dealer, st, i, int(id))
	}
	act, ok := vocabID(call)
	if !ok {
		t.Fatalf("bad call %q", call)
	}
	return matchFirst(entries, dealer, st.reader.calls, int(act), st)
}

const (
	revLine = "1X-1Y-2Z; X<Z = 5+X 4+Z p>16 F @2.0"
	penLine = "...-X; we_xx OR dbls_by_us>=2 OR gf OR their_bid>=4S = PENALTY"
	toLine  = "...-X = TO"
)

func TestPatternParse(t *testing.T) {
	e := mustEntries(t, revLine)[0]
	if e.pat != "1X-1Y-2Z" || e.wildcard || e.weight != 2.0 || e.artificial {
		t.Fatalf("reverse entry wrong: %+v", e)
	}
	if len(e.steps) != 3 || e.steps[0].them || e.steps[0].kind != stepBid ||
		e.steps[0].level != 1 || e.steps[0].denom != 'X' {
		t.Fatalf("steps wrong: %+v", e.steps)
	}
	if !e.flagF || e.flagGF || e.flagTO || e.flagPen {
		t.Errorf("flags wrong: %+v", e)
	}
	if len(e.branches) != 1 || len(e.branches[0]) != 4 ||
		e.branches[0][0].kind != tokLen || e.branches[0][0].suit != "X" ||
		e.branches[0][0].hi != 99 || e.branches[0][2].kind != tokPts ||
		e.branches[0][2].op != ">" || e.branches[0][3].kind != tokF {
		t.Errorf("tokens wrong: %+v", e.branches)
	}

	// the ">=" inside the clause must not be mistaken for the description "="
	e = mustEntries(t, penLine)[0]
	if !e.wildcard || !e.flagPen || len(e.atomGroups) != 4 {
		t.Fatalf("penalty entry wrong: %+v", e)
	}
	kinds := []int{e.atomGroups[0][0].kind, e.atomGroups[1][0].kind,
		e.atomGroups[2][0].kind, e.atomGroups[3][0].kind}
	want := []int{atomWeXX, atomCnt, atomGF, atomBidcmp}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("atom group %d kind = %d, want %d", i, kinds[i], want[i])
		}
	}
	if e.atomGroups[3][0].call != int(mustID(t, "4S")) || e.atomGroups[1][0].n != 2 {
		t.Errorf("atom args wrong: %+v %+v", e.atomGroups[3][0], e.atomGroups[1][0])
	}

	// parens, OR branches, attributes, ranges, no description
	e = mustEntries(t, "1X-(1Y)-2Z; Z<Y = 4+Z p>11.5 GF")[0]
	if !e.steps[1].them || !e.flagGF || e.weight != 1.0 {
		t.Errorf("competition entry wrong: %+v", e)
	}
	e = mustEntries(t, "1NT-2C = 4+H OR 4+S stayman ask @!")[0]
	if len(e.branches) != 2 || !e.artificial || e.steps[0].denom != 'N' {
		t.Errorf("stayman entry wrong: %+v", e)
	}
	e = mustEntries(t, "2X = 6-7X p4-12 weak two @! @0.1")[0]
	if e.branches[0][0].lo != 6 || e.branches[0][0].hi != 7 ||
		e.branches[0][1].op != "range" || e.weight != 0.1 {
		t.Errorf("weak two entry wrong: %+v", e)
	}
	e = mustEntries(t, "1H-2NT-3S")[0]
	if e.desc != "" || e.pat != "1H-2NT-3S" {
		t.Errorf("no-desc entry wrong: %+v", e)
	}
	for _, bad := range []string{
		"1Q-2X = nope",
		"1X = text @notanumber",
		"1X; nonsense_atom = text",
		"...-X = TO and PENALTY",
		"... = oops",
	} {
		if _, err := parseEntryLine(bad); err == nil {
			t.Errorf("parseEntryLine(%q) should fail", bad)
		}
	}
}

func mustID(t *testing.T, call string) uint8 {
	t.Helper()
	id, ok := vocabID(call)
	if !ok {
		t.Fatalf("bad call %q", call)
	}
	return id
}

// TestPatternMatchGoldens mirrors tests/test_patterns.py's matching cases;
// both sides must resolve identically.
func TestPatternMatchGoldens(t *testing.T) {
	rev := mustEntries(t, revLine)
	e, b := matchAt(t, rev, 0, []string{"1C", "P", "1H", "P"}, "2D")
	if e == nil || b['X'] != 0 || b['Y'] != 2 || b['Z'] != 1 {
		t.Fatalf("reverse should match with C<H bindings: %v %v", e, b)
	}
	// not a reverse: the new suit ranks below the opening
	if e, _ := matchAt(t, rev, 0, []string{"1H", "P", "1S", "P"}, "2D"); e != nil {
		t.Error("2D under 1H-1S is no reverse")
	}
	// opener's own-suit rebid
	if e, _ := matchAt(t, rev, 0, []string{"1C", "P", "1H", "P"}, "2C"); e != nil {
		t.Error("2C rebid matched the reverse")
	}
	// leading passes are skipped (3rd-seat opening)
	if e, _ := matchAt(t, rev, 0, []string{"P", "P", "1C", "P", "1H", "P"}, "2D"); e == nil {
		t.Error("reverse after two leading passes should match")
	}
	// opponents opened -> the anchored pattern must not fire
	if e, _ := matchAt(t, rev, 0, []string{"1H", "1S"}, "2C"); e != nil {
		t.Error("reverse matched with the opponents opening")
	}
	// dealer E: 1D(E)-P-1S(W)-P-2H(E) is a reverse for E-W (D < H)
	e, b = matchAt(t, rev, 1, []string{"1D", "P", "1S", "P"}, "2H")
	if e == nil || b['X'] != 1 || b['Y'] != 3 || b['Z'] != 2 {
		t.Fatalf("dealer-1 reverse wrong: %v %v", e, b)
	}

	// distinctness from literals and between variables
	lit := mustEntries(t, "1H-2X = artificial")
	if e, _ := matchAt(t, lit, 0, []string{"1H", "P"}, "2D"); e == nil {
		t.Error("2D over 1H should match 1H-2X")
	}
	if e, _ := matchAt(t, lit, 0, []string{"1H", "P"}, "2H"); e != nil {
		t.Error("variable bound to the pattern's literal suit")
	}
	raise := mustEntries(t, "1X-2X = raise")
	if e, _ := matchAt(t, raise, 0, []string{"1S", "P"}, "2S"); e == nil {
		t.Error("same letter should bind the same suit")
	}
	if e, _ := matchAt(t, raise, 0, []string{"1S", "P"}, "2D"); e != nil {
		t.Error("same letter bound two suits")
	}

	// fully literal sequences
	s := mustEntries(t, "1H-2N-3S")
	if e, _ := matchAt(t, s, 0, []string{"1H", "P", "2N", "P"}, "3S"); e == nil {
		t.Error("literal sequence should match")
	}
	if e, _ := matchAt(t, s, 0, []string{"1H", "P", "2N", "P"}, "3D"); e != nil {
		t.Error("literal sequence matched the wrong call")
	}

	// own passes are not skipped implicitly
	two := mustEntries(t, "1X-1Y = two responses")
	if e, _ := matchAt(t, two, 0, []string{"1C", "P", "P", "1D"}, "1S"); e != nil {
		t.Error("own pass between the bids was skipped")
	}
	if e, _ := matchAt(t, two, 0, []string{"1C", "P"}, "1S"); e == nil {
		t.Error("opponent pass should be skipped implicitly")
	}

	// wildcard tail vs anchored
	tail := mustEntries(t, "...-2N = anywhere")
	if e, _ := matchAt(t, tail, 0, []string{"1C", "P", "1D", "P", "1H", "P"}, "2N"); e == nil {
		t.Error("wildcard should match late in the auction")
	}
	open2n := mustEntries(t, "2N = opening")
	if e, _ := matchAt(t, open2n, 0, []string{"1C", "P", "1D", "P", "1H", "P"}, "2N"); e != nil {
		t.Error("anchored 2N matched late in the auction")
	}
	if e, _ := matchAt(t, open2n, 0, []string{"P", "P"}, "2N"); e == nil {
		t.Error("anchored 2N should match the opening")
	}

	// parenthesized steps match opponent calls only
	comp := mustEntries(t, "1X-(1Y)-2Z; Z<Y = GF")
	if e, _ := matchAt(t, comp, 0, []string{"1C", "1S"}, "2D"); e == nil {
		t.Error("2D under 1C-(1S) should match (D<S)")
	}
	if e, _ := matchAt(t, comp, 0, []string{"1C", "P"}, "2D"); e != nil {
		t.Error("matched without the opponent overcall")
	}
	if e, _ := matchAt(t, comp, 0, []string{"1C", "1D"}, "2H"); e != nil {
		t.Error("Z<Y violated but matched")
	}

	// first match wins in file order
	a := mustEntries(t, "...-X = first", "...-X = second")
	if e, _ := matchAt(t, a, 0, []string{"1H"}, "X"); e == nil || e.desc != "first" {
		t.Errorf("first-match-wins broken: %+v", e)
	}
}

func TestPatternDoubleConditions(t *testing.T) {
	dbls := mustEntries(t, penLine, toLine)
	kind := func(calls []string) string {
		e, _ := matchAt(t, dbls, 0, calls, "X")
		switch {
		case e == nil:
			return ""
		case e.flagPen:
			return "PENALTY"
		default:
			return "TO"
		}
	}
	if kind([]string{"1H"}) != "TO" {
		t.Error("plain double should be TO")
	}
	if kind([]string{"1H", "P", "P"}) != "TO" {
		t.Error("balancing double should still be TO")
	}
	if kind([]string{"P", "4S"}) != "PENALTY" {
		t.Error("double of their 4S should be PENALTY")
	}
	if kind([]string{"P", "4N"}) != "PENALTY" {
		t.Error("4N ranks above 4S")
	}
	if kind([]string{"P", "3S"}) != "TO" {
		t.Error("below 4S should stay TO")
	}
	if kind([]string{"1H", "X", "XX", "2D", "P", "P"}) != "PENALTY" {
		t.Error("our redouble should make it PENALTY")
	}
	if kind([]string{"1C", "1H", "X", "2D", "X", "2H"}) != "PENALTY" {
		t.Error("two prior doubles should make it PENALTY")
	}
	if kind([]string{"1C", "1H", "X", "2D", "P"}) != "TO" {
		t.Error("one prior double is not enough")
	}
	// file-defined GF: 1C-2D marked GF -> a later double is penalty
	all := mustEntries(t, "1X-2Y; Y>X = GF", penLine, toLine)
	e, _ := matchAt(t, all, 0, []string{"1C", "P", "2D", "2H", "3C", "3H"}, "X")
	if e == nil || !e.flagPen {
		t.Errorf("GF-conditioned PENALTY wrong: %+v", e)
	}
}

func TestPatternStateConditions(t *testing.T) {
	dbls := mustEntries(t, penLine, toLine)
	st := newMatchState(0)
	for i, c := range callIDs(t, []string{"1H", "P", "P"}) {
		foldState(dbls, 0, st, i, c)
	}
	if st.passesSinceBid != 2 || st.activity[1] != 0 {
		t.Fatalf("state wrong: %+v", st)
	}
	bal := mustEntries(t, "...-1S; balancing = balance seat")
	if e, _ := matchAt(t, bal, 0, []string{"1H", "P", "P"}, "1S"); e == nil {
		t.Error("balancing condition failed")
	}
	unc := mustEntries(t, "...-1S; uncontested = quiet")
	if e, _ := matchAt(t, unc, 0, []string{"1H", "P", "P"}, "1S"); e != nil {
		t.Error("uncontested matched a contested auction")
	}
	if e, _ := matchAt(t, unc, 0, []string{"P", "P"}, "1S"); e == nil {
		t.Error("uncontested should match after passes")
	}
	passed := mustEntries(t, "...-1S; passed = we passed")
	if e, _ := matchAt(t, passed, 0, []string{"P", "P", "1C", "P", "P"}, "1S"); e == nil {
		t.Error("passed-hand condition failed")
	}
	if e, _ := matchAt(t, passed, 0, []string{"1C", "P"}, "1S"); e != nil {
		t.Error("passed matched a hand that never called")
	}
	ourbid := mustEntries(t, "...-2D; our_bid>=1C = we opened")
	if e, _ := matchAt(t, ourbid, 0, []string{"1C", "1H"}, "2D"); e == nil {
		t.Error("our_bid condition failed")
	}
	if e, _ := matchAt(t, ourbid, 0, []string{"1H"}, "2D"); e != nil {
		t.Error("our_bid matched when we never bid")
	}
}

func TestPatternRenderDesc(t *testing.T) {
	rev := mustEntries(t, revLine)
	e, b := matchAt(t, rev, 0, []string{"1C", "P", "1H", "P"}, "2D")
	if got := renderDesc(e, b); got != "5+ ♣ 4+ ♦ 16+ pts forcing" {
		t.Errorf("reverse render: %q", got)
	}
	e, b = matchAt(t, mustEntries(t, "1NT-2C = 4+H OR 4+S stayman ask @! @1.0"),
		0, []string{"1N", "P"}, "2C")
	if got := renderDesc(e, b); got != "4+ ♥ or 4+ ♠ stayman ask" {
		t.Errorf("stayman render: %q", got)
	}
	e, _ = matchAt(t, mustEntries(t, "...-X = TO takeout"), 0, []string{"1H"}, "X")
	if got := renderDesc(e, nil); got != "takeout takeout" {
		t.Errorf("TO render: %q", got)
	}
	e, b = matchAt(t, mustEntries(t, "2X = 6-7X p4-12 weak two"), 0, nil, "2S")
	if got := renderDesc(e, b); got != "6-7 ♠ 4-12 pts weak two" {
		t.Errorf("weak two render: %q", got)
	}
}

func TestPatternAlertForCall(t *testing.T) {
	spec, err := parseSystem(`
[open_1nt_range]
alert = {call}: 15-17 balanced

1NT = p10-12 balanced @1.0
1H-2C =
`)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.entries) != 2 {
		t.Fatalf("want 2 entries, got %d", len(spec.entries))
	}
	// the pattern entry wins over the section
	alerts := spec.callAlerts(0, []string{"1N"}, 0)
	a, _ := alerts[0].(*bidAlert)
	if a == nil || a.Text != "10-12 pts balanced" {
		t.Fatalf("pattern should win: %+v", alerts[0])
	}
	// an entry with an empty description suppresses the alert (and the
	// section fallback for that call)
	spec2, err := parseSystem("[two_one_gf]\nalert = {call}: 2/1\n\n1H-2C = @0")
	if err != nil {
		t.Fatal(err)
	}
	if got := spec2.callAlerts(0, []string{"1H", "P", "2C"}, 0); got != nil && got[2] != nil {
		t.Errorf("empty description should suppress: %+v", got[2])
	}
	// no pattern match -> section fallback still works
	spec3, _ := parseSystem("[open_1nt_range]\nalert = {call}: 15-17 balanced\n\n1H-2C = x")
	alerts = spec3.callAlerts(0, []string{"1N"}, 0)
	a, _ = alerts[0].(*bidAlert)
	if a == nil || a.Text != "1NT: 15-17 balanced" {
		t.Fatalf("section fallback broken: %+v", alerts[0])
	}
}

func TestPatternKVRouting(t *testing.T) {
	// bare-identifier lines stay key = value even when unknown to Go;
	// call tokens and pattern-shaped lines become entries
	spec, err := parseSystem("[general]\nkr_tolerance = 0.5\n\nP = a passed-hand note\n2H = weak")
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.entries) != 2 {
		t.Fatalf("want 2 entries, got %d", len(spec.entries))
	}
	if _, err := parseSystem("[general]\nkr_tolerance\n"); err == nil ||
		!strings.Contains(err.Error(), "expected key = value") {
		t.Errorf("missing = should fail as a kv line: %v", err)
	}
	if _, err := parseSystem("nt_range = 15-17\n"); err == nil ||
		!strings.Contains(err.Error(), "outside any section") {
		t.Errorf("kv outside a section should fail: %v", err)
	}
}
