package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// miniSystem is a tiny stand-in for models/system.txt used by the tests.
const miniSystem = `
# a tiny system file
[general]
kr_tolerance = 0.5

[open_1nt_range]
alert = {call}: 12-14 balanced
weight = 2.0
nt_range = 12-14

[open_weak2]
alert = {call}: weak two, 6-card {suit}
artificial = yes
weight = 0.1

[rkcb_response]
alert = {call}: RKCB for {trump} -- {shows}
artificial = yes
steps = zero or three | one or four | two without | two with

[splinter]
alert = {call}: splinter, short {suit}, fit for partner's {partner_suit}
artificial = yes

[overcall]
alert = {call}: overcall, 5+ {suit}
weight = 1.0

[smolen]
alert = {call}: Smolen, 4 {suit} and 5+ {other}
artificial = yes

[gf_pass]
weight = 2.0
`

func writeMiniSystem(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "system.txt")
	if err := os.WriteFile(path, []byte(miniSystem), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestVocabID(t *testing.T) {
	for call, want := range map[string]uint8{"P": 1, "X": 2, "XX": 3, "1C": 4, "1D": 5, "1N": 8, "3S": 17, "7N": 38} {
		got, ok := vocabID(call)
		if !ok || got != want {
			t.Errorf("vocabID(%q) = %d, %v; want %d, true", call, got, ok, want)
		}
	}
	for _, bad := range []string{"", "PAD", "8C", "1T", "pass", "NT"} {
		if _, ok := vocabID(bad); ok {
			t.Errorf("vocabID(%q) should fail", bad)
		}
	}
}

func TestParseSystem(t *testing.T) {
	spec, err := parseSystem(miniSystem)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.sections) != 8 || spec.nAlerts != 6 {
		t.Fatalf("got %d sections, %d alerts", len(spec.sections), spec.nAlerts)
	}
	s := spec.sections["open_1nt_range"]
	if s.alert != "{call}: 12-14 balanced" || s.artificial {
		t.Errorf("open_1nt_range wrong: %+v", s)
	}
	if !spec.sections["open_weak2"].artificial {
		t.Error("open_weak2 should be artificial")
	}
	if got := spec.sections["rkcb_response"].steps; len(got) != 4 || got[1] != "one or four" {
		t.Errorf("rkcb steps wrong: %+v", got)
	}
	if spec.sections["gf_pass"].alert != "" {
		t.Error("training-only sections carry no alert")
	}
	// training-side keys are ignored, not errors
	if _, ok := spec.sections["open_1nt_range"]; !ok {
		t.Fatal("section missing")
	}
}

func TestParseSystemErrors(t *testing.T) {
	for _, bad := range []string{
		"[general\n",                  // unterminated header
		"kr_tolerance = 1\n",          // key outside a section
		"[general]\nno equals here\n", // malformed line
		"[general]\n[general]\n",      // duplicate section
		"# only comments\n",           // nothing parsed
	} {
		if _, err := parseSystem(bad); err == nil {
			t.Errorf("parseSystem(%q) should fail", bad)
		}
	}
}

// classifyNext feeds the auction (dealer-relative) and classifies `call` for
// the seat whose turn it is, mirroring the Python tests' helper.
func classifyNext(t *testing.T, calls []string, call string) callInfo {
	t.Helper()
	r := newAuctionReader(0)
	for _, c := range calls {
		id, ok := vocabID(c)
		if !ok {
			t.Fatalf("bad call %q in %v", c, calls)
		}
		r.feed(int(id))
	}
	id, ok := vocabID(call)
	if !ok {
		t.Fatalf("bad call %q", call)
	}
	return r.classify(int(id), r.nextSeat())
}

// TestSectionFor mirrors tests/test_system_file.py::test_section_for: the
// Go resolver must agree with the Python one on every case.
func TestSectionFor(t *testing.T) {
	P := "P"
	cases := []struct {
		calls   []string
		call    string
		section string
	}{
		// openings
		{nil, "1N", "open_1nt_range"},
		{nil, "2N", "open_2nt_range"},
		{nil, "2C", "open_2c_strong"},
		{nil, "2H", "open_weak2"},
		{nil, "1S", "open_major_5card"},
		{nil, "1D", "open_minor_length"},
		{nil, "3S", ""}, // preempt: undefined
		{[]string{P, P, P}, "1C", "open_minor_length"},
		// strong 2C structure
		{[]string{"2C", P}, "2D", "resp_2c_waiting"},
		{[]string{"2C", P}, "2H", "resp_2c_positive"},
		{[]string{"2C", P}, "2N", "resp_2c_positive"},
		// responses / raises
		{[]string{"1H", P}, "1N", "resp_1nt"},
		{[]string{"1H", P}, "2H", "resp_raise"},
		{[]string{"1H", P}, "3H", "resp_jump_raise"},
		{[]string{"1H", P}, "2N", "resp_2nt"},
		{[]string{"1H", P, "2C", P, "2D", P}, "3N", ""}, // responder already called
		{[]string{"1C", P}, "3N", "resp_3nt"},
		{[]string{"1H", P}, "2C", "two_one_gf"},
		{[]string{"1S", P}, "2C", "two_one_gf"},
		// opener rebids
		{[]string{"1C", P, "1H", P}, "1N", "rebid_1nt"},
		{[]string{"1C", P, "1H", P}, "2N", "rebid_2nt"},
		{[]string{"1C", P, "1H", P}, "3D", "strong_jump_rebid"},
		{[]string{"1C", P, "1H", P}, "2C", ""}, // own rebid: natural
		// XYZ
		{[]string{"1C", P, "1D", P, "1H", P}, "2C", "xyz_2c"},
		{[]string{"1C", P, "1D", P, "1H", P}, "2D", "xyz_2d"},
		{[]string{"1C", P, "1D", P, "1H", P, "2C", P}, "2D", "xyz_relay"},
		{[]string{"1C", P, "1D", P, "1H", P, "2C", P, "2D", P}, P, "xyz_pass"},
		{[]string{"1C", P, "1D", "1S", "1H", P}, "2C", ""}, // contested: no XYZ
		// smolen
		{[]string{"1N", P, "2C", P, "2D", P}, "3H", "smolen"},
		{[]string{"2N", P, "3C", P, "3D", P}, "3S", "smolen"},
		// RKCB
		{[]string{"1S", P, "2S", P}, "4N", "rkcb_ask"},
		{[]string{"1S", P, "2S", P, "4N", P}, "5C", "rkcb_response"},
		{[]string{"1S", P, "2S", P, "4N", P}, "5D", "rkcb_response"},
		// splinter / jump shift / fourth suit / cuebid
		{[]string{"1S", P}, "4D", "splinter"},
		{[]string{"1C", "1D"}, "2H", "inv_jump_shift"},
		{[]string{"1C", P, "1H", P, "2D", P}, "2S", "fourth_suit"}, // 1C-1H-2D: no XYZ
		{[]string{"1C", P, "1H", P, "1S", P}, "2D", "xyz_2d"},      // 1x-1y-1z: XYZ wins
		{[]string{"1H", "1S"}, "2S", "cuebid_support"},
		// competition
		{[]string{"1H"}, "1S", "overcall"},
		{[]string{"1H"}, "1N", "overcall_1nt"},
		{[]string{"1H", P, P}, "1N", "overcall_1nt_bal"},
		{[]string{"1H", P, P}, "1S", "overcall"},
		{[]string{"1H"}, "X", "takeout_double"},
		{[]string{"1H", P, P}, "X", "balancing_takeout"},
		{[]string{"1C", "1H"}, "X", "negative_double"},
		{[]string{"1H", "1S", "2C", P, "2D"}, "X", "penalty_double"},
		{[]string{"1N"}, "X", "penalty_double"}, // nt_penalty folds into penalty
		{nil, P, ""},
		{nil, "XX", ""},
	}
	for _, c := range cases {
		info := classifyNext(t, c.calls, c.call)
		if got := sectionFor(info); got != c.section {
			t.Errorf("sectionFor(%v -> %s) = %q, want %q", c.calls, c.call, got, c.section)
		}
	}
}

func TestRenderAlert(t *testing.T) {
	spec, err := parseSystem(miniSystem)
	if err != nil {
		t.Fatal(err)
	}
	// {call} and {suit}
	a := spec.alertFor(classifyNext(t, nil, "2S"))
	if a == nil || !strings.HasPrefix(a.Text, "2♠: weak two, 6-card ♠") || !a.Artificial {
		t.Fatalf("weak two alert wrong: %+v", a)
	}
	// splinter: {suit} + {partner_suit}
	a = spec.alertFor(classifyNext(t, []string{"1S", "P"}, "4D"))
	if a == nil || !strings.Contains(a.Text, "short ♦") || !strings.Contains(a.Text, "partner's ♠") {
		t.Fatalf("splinter alert wrong: %+v", a)
	}
	// smolen: {suit} + {other}
	a = spec.alertFor(classifyNext(t, []string{"1N", "P", "2C", "P", "2D", "P"}, "3H"))
	if a == nil || !strings.Contains(a.Text, "4 ♥") || !strings.Contains(a.Text, "5+ ♠") {
		t.Fatalf("smolen alert wrong: %+v", a)
	}
	// rkcb response: {trump} + {shows} from the steps line
	a = spec.alertFor(classifyNext(t, []string{"1S", "P", "2S", "P", "4N", "P"}, "5D"))
	if a == nil || !strings.Contains(a.Text, "RKCB for ♠ -- one or four") {
		t.Fatalf("rkcb alert wrong: %+v", a)
	}
	// a non-artificial section renders without the marker
	if a := spec.alertFor(classifyNext(t, nil, "1N")); a == nil || a.Text != "1NT: 12-14 balanced" || a.Artificial {
		t.Fatalf("1NT alert wrong: %+v", a)
	}
}

func TestAlertPayloads(t *testing.T) {
	spec, err := parseSystem(miniSystem)
	if err != nil {
		t.Fatal(err)
	}
	// dealer East (1): call 0 = E (bot), call 1 = S (human seat 2, masked)
	alerts := spec.callAlerts(1, []string{"2S", "P"}, 1<<2)
	if len(alerts) != 2 {
		t.Fatalf("want 2 entries, got %d", len(alerts))
	}
	if alerts[0] == nil || !alerts[0].(*bidAlert).Artificial {
		t.Errorf("bot weak two should carry an artificial alert: %+v", alerts[0])
	}
	if alerts[1] != nil {
		t.Error("human-seat call should be nil")
	}
	// an unknown call aborts the walk and leaves the tail nil
	alerts = spec.callAlerts(1, []string{"2S", "BOGUS"}, 0)
	if alerts[0] == nil || alerts[1] != nil {
		t.Errorf("unexpected tail handling: %+v", alerts)
	}
	// calls with no meaning in the file collapse to nil
	if alerts := spec.callAlerts(0, []string{"P", "P"}, 0); alerts != nil {
		t.Errorf("passes should produce no alerts: %+v", alerts)
	}
	// options at the root, parallel to legal (only sections the file covers)
	opts := spec.optionAlerts(0, nil, []string{"P", "1N", "2H", "2S"})
	if opts == nil || opts[0] != nil || opts[1] == nil || opts[2] == nil || opts[3] == nil {
		t.Fatalf("unexpected options: %+v", opts)
	}
	// JSON shape: text, artificial only when set
	b, _ := json.Marshal(opts[1])
	if string(b) != `{"text":"1NT: 12-14 balanced"}` {
		t.Errorf("unexpected JSON: %s", b)
	}
	b, _ = json.Marshal(opts[2])
	if !strings.Contains(string(b), `"artificial":true`) || !strings.Contains(string(b), "weak two") {
		t.Errorf("artificial missing from JSON: %s", b)
	}
	// options in a later position use the dealer-relative seats
	opts = spec.optionAlerts(0, []string{"1S", "P"}, []string{"4D"})
	if opts == nil || opts[0] == nil || !strings.Contains(opts[0].(*bidAlert).Text, "splinter") {
		t.Fatalf("splinter option missing: %+v", opts)
	}
}

// TestStateLockedAlerts checks the board payload wiring: alerts parallel to
// the calls made (bot seats only) and option alerts parallel to legal.
func TestStateLockedAlerts(t *testing.T) {
	oldPath := sysPath
	sysPath = writeMiniSystem(t)
	defer func() {
		sysPath = oldPath
		sysState.mu.Lock()
		sysState.spec, sysState.mod, sysState.size = nil, time.Time{}, 0
		sysState.mu.Unlock()
	}()
	b := &PlayBoard{
		ID: "x", User: "u", Dealer: 1, Vuln: 0,
		Calls:    []string{"2H"}, // East (bot) opened a weak two; South to call
		Legal:    []string{"P", "2S"},
		LegalLen: 1,
		Created:  time.Now(),
	}
	st := b.stateLocked(PlayStats{})
	alerts, ok := st["alerts"].([]any)
	if !ok || len(alerts) != 1 || alerts[0] == nil {
		t.Fatalf("alerts missing or wrong: %+v", st["alerts"])
	}
	if !strings.Contains(alerts[0].(*bidAlert).Text, "weak two") {
		t.Errorf("alert text wrong: %+v", alerts[0])
	}
	opts, ok := st["optionAlerts"].([]any)
	if !ok || len(opts) != 2 || opts[0] != nil || opts[1] == nil {
		t.Fatalf("optionAlerts missing or wrong: %+v", st["optionAlerts"])
	}
	if !strings.Contains(opts[1].(*bidAlert).Text, "overcall") {
		t.Errorf("option text wrong: %+v", opts[1])
	}
	// human-seat calls carry no alert: South (2) deals and opens
	b2 := &PlayBoard{
		ID: "y", User: "u", Dealer: 2, Vuln: 0,
		Calls: []string{"2H"}, LegalLen: -1, Created: time.Now(),
	}
	if a := b2.stateLocked(PlayStats{})["alerts"]; a != nil {
		t.Errorf("human bid should not produce alerts: %+v", a)
	}
}

// TestSystemReadyReload checks the mtime refresh and the missing-file path.
func TestSystemReadyReload(t *testing.T) {
	oldPath := sysPath
	path := writeMiniSystem(t)
	sysPath = path
	defer func() {
		sysPath = oldPath
		sysState.mu.Lock()
		sysState.spec, sysState.mod, sysState.size = nil, time.Time{}, 0
		sysState.mu.Unlock()
	}()
	spec := systemReady()
	if spec == nil || len(spec.sections) != 8 {
		t.Fatalf("first load failed: %+v", spec)
	}
	if systemReady() != spec {
		t.Error("unchanged file should reuse the cached spec")
	}
	// rewrite with a different mtime -> reload
	if err := os.WriteFile(path, []byte(miniSystem+"\n[resp_1nt]\nalert = {call}: 5-12\n"), 0644); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}
	spec2 := systemReady()
	if spec2 == spec || spec2 == nil || len(spec2.sections) != 9 {
		t.Fatalf("reload did not happen: %+v", spec2)
	}
	// missing file -> nil (boards served without alerts)
	sysPath = filepath.Join(t.TempDir(), "gone.txt")
	if systemReady() != nil {
		t.Error("missing file should yield nil")
	}
}

// TestRealSystemFile parses the shipped models/system.txt (skipped when
// absent) and spot-checks a few alerts against it.
func TestRealSystemFile(t *testing.T) {
	path := systemFilePath()
	if _, err := os.Stat(path); err != nil {
		t.Skip("no system file:", err)
	}
	spec, err := loadSystemFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.sections) < 30 || spec.nAlerts < 30 {
		t.Fatalf("implausible system file: %d sections, %d alerts", len(spec.sections), spec.nAlerts)
	}
	for _, want := range []string{
		"open_1nt_range", "open_2c_strong", "open_weak2", "xyz_2c", "xyz_2d",
		"xyz_relay", "xyz_pass", "rkcb_ask", "rkcb_response", "splinter",
		"smolen", "fourth_suit", "cuebid_support", "two_one_gf", "gf_pass",
	} {
		if _, ok := spec.sections[want]; !ok {
			t.Errorf("shipped system file lacks [%s]", want)
		}
	}
	a := spec.alertFor(classifyNext(t, nil, "1N"))
	if a == nil || !strings.Contains(a.Text, "15-17") {
		t.Errorf("shipped 1NT alert wrong: %+v", a)
	}
	a = spec.alertFor(classifyNext(t, []string{"1S", "P", "2S", "P", "4N", "P"}, "5C"))
	if a == nil || !strings.Contains(a.Text, "0 or 3 keycards") || !a.Artificial {
		t.Errorf("shipped RKCB response alert wrong: %+v", a)
	}
	// the shipped pattern entries: reverse, stayman, conditional doubles
	if len(spec.entries) < 8 {
		t.Errorf("shipped file carries %d pattern entries, want >= 8", len(spec.entries))
	}
	alerts := spec.callAlerts(0, []string{"1C", "P", "1H", "P", "2D"}, 0)
	rev, _ := alerts[4].(*bidAlert)
	if rev == nil || !strings.Contains(rev.Text, "reverse: 5+ ♣ 4+ ♦") ||
		!strings.Contains(rev.Text, "forcing") || strings.Contains(rev.Text, "{") {
		t.Errorf("shipped reverse alert wrong: %+v", alerts[4])
	}
	alerts = spec.callAlerts(0, []string{"1N", "P", "2C"}, 0)
	stay, _ := alerts[2].(*bidAlert)
	if stay == nil || !stay.Artificial || !strings.Contains(stay.Text, "4+ ♥ or 4+ ♠") {
		t.Errorf("shipped stayman alert wrong: %+v", alerts[2])
	}
	// plain double of their opening: takeout
	alerts = spec.callAlerts(0, []string{"1H", "X"}, 0)
	tbl, _ := alerts[1].(*bidAlert)
	if tbl == nil || !strings.Contains(tbl.Text, "takeout") {
		t.Errorf("shipped TO double alert wrong: %+v", alerts[1])
	}
	// double of their 4S: penalty
	alerts = spec.callAlerts(0, []string{"P", "4S", "X"}, 0)
	pen, _ := alerts[2].(*bidAlert)
	if pen == nil || !strings.HasPrefix(pen.Text, "penalty") {
		t.Errorf("shipped PENALTY double alert wrong: %+v", alerts[2])
	}
	// every rendered alert must be placeholder-free for a sample auction
	for _, calls := range [][]string{
		{"1N"}, {"2C"}, {"2H"}, {"1S", "P", "2S"},
		{"1C", "P", "1D", "P", "1H", "P", "2C"},
		{"1N", "P", "2C", "P", "2D", "P", "3S"},
		{"1S", "P", "2S", "P", "4N", "P", "5D"},
		{"1H", "1S", "2S"}, {"1H", "P", "P", "1N"},
		{"1C", "P", "1H", "P", "2D"}, {"1H", "X", "P", "P"},
		{"P", "4S", "X", "P", "5C"},
	} {
		for _, entry := range spec.callAlerts(0, calls, 0) {
			if a, ok := entry.(*bidAlert); ok && strings.Contains(a.Text, "{") {
				t.Errorf("unfilled placeholder for %v: %s", calls, a.Text)
			}
		}
	}
}
