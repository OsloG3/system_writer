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
kr_tolerance = 0.5

1X-1Y-2Z; X<Z = reverse: 5+X 4+Z p>16 F @2.0
2X = 6-7X weak two @! @0.1
1N = p12-14 balanced
...-2Z; !we_bid, notopening = overcall: 5+Z
...-X; their_bid>=4S = PENALTY -- partner pulls only with shortness
...-X = TO -- partner passes only with a trump stack
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
	if len(spec.entries) != 6 || spec.nAlerts != 6 {
		t.Fatalf("got %d entries, %d alerts", len(spec.entries), spec.nAlerts)
	}
	e := spec.entries[0]
	if e.pat != "1X-1Y-2Z" || !e.anchored || e.weight != 2.0 || !e.flagF {
		t.Errorf("reverse entry wrong: %+v", e)
	}
	e = spec.entries[1]
	if !e.artificial || e.weight != 0.1 || e.artificial != true {
		t.Errorf("weak two entry wrong: %+v", e)
	}
	if !spec.entries[4].flagPen || spec.entries[4].anchored {
		t.Errorf("penalty entry wrong: %+v", spec.entries[4])
	}
	if !spec.entries[5].flagTO {
		t.Errorf("TO entry wrong: %+v", spec.entries[5])
	}
	// config lines are accepted (and ignored) -- kr_tolerance above
}

func TestParseSystemErrors(t *testing.T) {
	for _, bad := range []string{
		"[general]\nkr_tolerance = 1.0\n", // INI sections are gone
		"kr_tolerance\n",                  // no "=" on a config-shaped line
		"1Q = nope\n",                     // bad denom
		"1X; bogus_atom = text\n",         // bad clause atom
		"# only comments\n",               // nothing parsed
	} {
		if _, err := parseSystem(bad); err == nil {
			t.Errorf("parseSystem(%q) should fail", bad)
		}
	}
}

func TestAlertPayloads(t *testing.T) {
	spec, err := parseSystem(miniSystem)
	if err != nil {
		t.Fatal(err)
	}
	// dealer East (1): call 0 = E (bot) opens a weak two, call 1 = S (human,
	// masked)
	alerts := spec.callAlerts(1, []string{"2S", "P"}, 1<<2)
	if len(alerts) != 2 {
		t.Fatalf("want 2 entries, got %d", len(alerts))
	}
	a, _ := alerts[0].(*bidAlert)
	if a == nil || !a.Artificial || !strings.Contains(a.Text, "6-7 ♠") {
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
	// options at the root, parallel to legal
	opts := spec.optionAlerts(0, nil, []string{"P", "1N", "2H", "2S"})
	if opts == nil || opts[0] != nil || opts[1] == nil || opts[2] == nil || opts[3] == nil {
		t.Fatalf("unexpected options: %+v", opts)
	}
	// JSON shape: text, artificial only when set
	b, _ := json.Marshal(opts[1])
	if string(b) != `{"text":"12-14 pts balanced"}` {
		t.Errorf("unexpected JSON: %s", b)
	}
	b, _ = json.Marshal(opts[2])
	if !strings.Contains(string(b), `"artificial":true`) || !strings.Contains(string(b), "weak two") {
		t.Errorf("artificial missing from JSON: %s", b)
	}
	// options in a later position use the dealer-relative seats
	opts = spec.optionAlerts(0, []string{"1C", "P", "1H", "P"}, []string{"2D"})
	if opts == nil || opts[0] == nil || !strings.Contains(opts[0].(*bidAlert).Text, "reverse") {
		t.Fatalf("reverse option missing: %+v", opts)
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
	if spec == nil || len(spec.entries) != 6 {
		t.Fatalf("first load failed: %+v", spec)
	}
	if systemReady() != spec {
		t.Error("unchanged file should reuse the cached spec")
	}
	// rewrite with a different mtime -> reload
	if err := os.WriteFile(path, []byte(miniSystem+"\n1H-2C = checkback\n"), 0644); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}
	spec2 := systemReady()
	if spec2 == spec || spec2 == nil || len(spec2.entries) != 7 {
		t.Fatalf("reload did not happen: %+v", spec2)
	}
	// missing file -> nil (boards served without alerts)
	sysPath = filepath.Join(t.TempDir(), "gone.txt")
	if systemReady() != nil {
		t.Error("missing file should yield nil")
	}
}

// TestRealSystemFile parses the shipped models/system.txt (skipped when
// absent) and spot-checks the alerts it generates.
func TestRealSystemFile(t *testing.T) {
	path := systemFilePath()
	if _, err := os.Stat(path); err != nil {
		t.Skip("no system file:", err)
	}
	spec, err := loadSystemFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.entries) < 60 {
		t.Fatalf("implausible system file: %d entries", len(spec.entries))
	}
	type spot struct {
		calls []string
		call  string
		want  string
		art   bool
	}
	spots := []spot{
		{nil, "1N", "15-17 pts balanced", false},
		{nil, "2H", "weak two", true},
		{nil, "2C", "strong artificial", true},
		{nil, "3S", "preempt", false},
		{[]string{"1C", "P", "1H", "P"}, "2D", "reverse 5+ ♣ 4+ ♦ 16+ pts forcing", false},
		{[]string{"1N", "P"}, "2C", "4+ ♥ or 4+ ♠ stayman ask", true},
		{[]string{"2C", "P"}, "2D", "waiting", true},
		{[]string{"1C", "P", "1D", "P", "1H", "P"}, "2C", "XYZ relay", true},
		{[]string{"1S", "P", "2S", "P", "4N", "P"}, "5C", "0 or 3 keycards", true},
		{[]string{"1S", "P", "2S", "P"}, "4N", "RKCB ask for ♠", true},
		{[]string{"1S", "P"}, "4D", "splinter in ♦", true},
		{[]string{"1H", "P"}, "2H", "raise 3+ ♥", false},
		{[]string{"1H", "P"}, "2C", "2/1 3+ ♣", false},
		{[]string{"1C", "1S"}, "2D", "2/1 over their bid", false},
		{[]string{"1H"}, "X", "takeout", false},
		{[]string{"P", "4S"}, "X", "penalty", false},
		{[]string{"1H", "P", "P"}, "1S", "balancing overcall", false},
		{[]string{"1N", "P", "2C", "P", "2D", "P"}, "3S", "smolen", true},
		{[]string{"1C", "P", "1H", "P"}, "2N", "opener's jump 2NT rebid", false},
		{[]string{"1H", "P"}, "1N", "no biddable 4-card major", false},
		{[]string{"1C", "P", "1H", "P", "2D", "P"}, "2S", "fourth suit", true},
	}
	for _, s := range spots {
		got := firstAlertText(spec, 0, s.calls, s.call)
		if got == nil || !strings.Contains(got.Text, s.want) {
			t.Errorf("%v -> %s: got %+v, want text containing %q",
				s.calls, s.call, got, s.want)
			continue
		}
		if got.Artificial != s.art {
			t.Errorf("%v -> %s: artificial = %v, want %v (%q)",
				s.calls, s.call, got.Artificial, s.art, got.Text)
		}
		if strings.Contains(got.Text, "{") {
			t.Errorf("%v -> %s: unfilled placeholder: %s", s.calls, s.call, got.Text)
		}
	}
	// uncovered calls carry no alert
	if a := firstAlertText(spec, 0, []string{"P"}, "P"); a != nil {
		t.Errorf("plain pass should have no alert: %+v", a)
	}
	if a := firstAlertText(spec, 0, []string{"1C", "P", "1H", "P"}, "2C"); a != nil {
		t.Errorf("opener's own-suit rebid should have no alert: %+v", a)
	}
}

// firstAlertText returns the alert for `call` after `calls` (dealer-relative),
// rendered through the same walk the board payloads use.
func firstAlertText(spec *systemSpec, dealer int, calls []string, call string) *bidAlert {
	st := newMatchState(dealer)
	for i, c := range calls {
		id, ok := vocabID(c)
		if !ok {
			return nil
		}
		foldState(spec.entries, dealer, st, i, int(id))
	}
	id, ok := vocabID(call)
	if !ok {
		return nil
	}
	return spec.alertForCall(st, int(id))
}
