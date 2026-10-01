package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

const tinyReport = `{
  "ckpt": "../models/last.pt", "deals": 100, "seed": 0, "greedy": false, "temp": 1.0,
  "tree": {
    "call": null, "n": 100, "hcp": null, "suits": null,
    "children": {
      "P": {"call": "P", "n": 50, "hcp": [0, 12],
            "suits": {"S": [0, 6], "H": [0, 5], "D": [0, 6], "C": [0, 6]},
            "children": {
              "1C": {"call": "1C", "n": 20, "hcp": [10, 15],
                     "suits": {"S": [1, 5], "H": [0, 4], "D": [1, 5], "C": [2, 7]},
                     "children": {}}
            }},
      "1C": {"call": "1C", "n": 40, "hcp": [12, 20],
             "suits": {"S": [0, 7], "H": [0, 7], "D": [0, 7], "C": [2, 8]},
             "children": {}},
      "1N": {"call": "1N", "n": 10, "hcp": [14, 16],
             "suits": {"S": [2, 5], "H": [2, 5], "D": [2, 5], "C": [2, 5]},
             "children": {}}
    }
  }
}`

func writeTiny(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tree.json")
	if err := os.WriteFile(path, []byte(tinyReport), 0644); err != nil {
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

func TestCompileTiny(t *testing.T) {
	tr, err := compileAlertTree(writeTiny(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.nodes) != 5 || tr.deals != 100 {
		t.Fatalf("got %d nodes, deals %d; want 5 nodes, deals 100", len(tr.nodes), tr.deals)
	}
	if tr.nodes[0].seen != 0 {
		t.Error("root should carry no hand stats")
	}
	// walk P -> 1C
	p := tr.child(0, "P")
	pc := tr.child(p, "1C")
	if pc < 0 {
		t.Fatal("P -> 1C not found")
	}
	a := tr.alertAt(p, pc)
	if a == nil || a.N != 20 || a.HCP != [2]int16{10, 15} || a.Suits[3] != [2]int8{2, 7} || a.Pct != 0.4 {
		t.Fatalf("unexpected alert for P,1C: %+v", a)
	}
	if tr.walk([]string{"P", "1C"}) != pc || tr.walk([]string{"P", "X"}) != -1 {
		t.Error("walk mismatch")
	}
}

func TestAlertPayloads(t *testing.T) {
	tr, err := compileAlertTree(writeTiny(t))
	if err != nil {
		t.Fatal(err)
	}
	// dealer East (1): call 0 = E (bot), call 1 = S (human seat 2, masked)
	alerts := tr.callAlerts(1, []string{"P", "1C"}, 1<<2)
	if len(alerts) != 2 {
		t.Fatalf("want 2 entries, got %d", len(alerts))
	}
	if alerts[0] == nil {
		t.Error("bot call should carry an alert")
	}
	if alerts[1] != nil {
		t.Error("human-seat call should be nil")
	}
	// an unknown call aborts the walk and leaves the tail nil
	alerts = tr.callAlerts(1, []string{"P", "7N"}, 0)
	if alerts[0] == nil || alerts[1] != nil {
		t.Errorf("unexpected tail handling: %+v", alerts)
	}
	// options at the root, parallel to legal
	opts := tr.optionAlerts(nil, []string{"P", "1C", "1N", "2H"})
	if opts == nil || opts[0] == nil || opts[1] == nil || opts[2] == nil || opts[3] != nil {
		t.Fatalf("unexpected options: %+v", opts)
	}
	first := opts[1].(*bidAlert)
	if first.N != 40 || first.Pct != 0.4 {
		t.Errorf("root 1C: %+v", first)
	}
	// a position the tree never saw has no options
	if opts := tr.optionAlerts([]string{"7N"}, []string{"P"}); opts != nil {
		t.Errorf("want nil for unseen position, got %+v", opts)
	}
	// JSON shape stays compact and ordered
	b, _ := json.Marshal(opts[0])
	if string(b) != `{"n":50,"pct":0.5,"hcp":[0,12],"suits":[[0,6],[0,5],[0,6],[0,6]]}` {
		t.Errorf("unexpected JSON: %s", b)
	}
}

func TestBinRoundtrip(t *testing.T) {
	path := writeTiny(t)
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	want, err := compileAlertTree(path)
	if err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(t.TempDir(), "tree.bin")
	if err := writeAlertBin(bin, want, fi); err != nil {
		t.Fatal(err)
	}
	got, err := readAlertBin(bin, fi)
	if err != nil {
		t.Fatal(err)
	}
	if got.deals != want.deals || len(got.nodes) != len(want.nodes) || len(got.kids) != len(want.kids) {
		t.Fatal("cache roundtrip changed the tree shape")
	}
	for i := range want.nodes {
		if got.nodes[i] != want.nodes[i] {
			t.Fatalf("node %d differs: %+v vs %+v", i, got.nodes[i], want.nodes[i])
		}
	}
	for i := range want.kids {
		if got.kids[i] != want.kids[i] {
			t.Fatalf("kid %d differs", i)
		}
	}
	// touching the source invalidates the cache
	newFi := fakeFileInfo{size: fi.Size() + 1, mod: fi.ModTime().Add(time.Second)}
	if _, err := readAlertBin(bin, newFi); err == nil {
		t.Error("stale cache was accepted")
	}
}

type fakeFileInfo struct {
	size int64
	mod  time.Time
}

// TestStateLockedAlerts checks the board payload wiring: alerts parallel to
// the calls made (bot seats only) and option alerts parallel to legal.
func TestStateLockedAlerts(t *testing.T) {
	tr, err := compileAlertTree(writeTiny(t))
	if err != nil {
		t.Fatal(err)
	}
	alertTreePtr.Store(tr)
	defer alertTreePtr.Store(nil)
	b := &PlayBoard{
		ID: "x", User: "u", Dealer: 1, Vuln: 0,
		Calls:    []string{"P"}, // East (bot) passed
		Legal:    []string{"P", "1C"},
		LegalLen: 1,
		Created:  time.Now(),
	}
	st := b.stateLocked(PlayStats{})
	alerts, ok := st["alerts"].([]any)
	if !ok || len(alerts) != 1 || alerts[0] == nil {
		t.Fatalf("alerts missing or wrong: %+v", st["alerts"])
	}
	opts, ok := st["optionAlerts"].([]any)
	if !ok || len(opts) != 2 || opts[0] != nil || opts[1] == nil {
		t.Fatalf("optionAlerts missing or wrong: %+v", st["optionAlerts"])
	}
	// human-seat calls carry no alert: South (2) deals and passes
	b2 := &PlayBoard{
		ID: "y", User: "u", Dealer: 2, Vuln: 0,
		Calls: []string{"P"}, LegalLen: -1, Created: time.Now(),
	}
	if a := b2.stateLocked(PlayStats{})["alerts"]; a != nil {
		t.Errorf("human pass should not produce alerts: %+v", a)
	}
}

func (f fakeFileInfo) Name() string       { return "tree.json" }
func (f fakeFileInfo) Size() int64        { return f.size }
func (f fakeFileInfo) Mode() os.FileMode  { return 0644 }
func (f fakeFileInfo) ModTime() time.Time { return f.mod }
func (f fakeFileInfo) IsDir() bool        { return false }
func (f fakeFileInfo) Sys() any           { return nil }

// TestCompressedSources feeds the same report through zstd and gzip and
// checks the compiled trees match the plain-JSON one, plus bin-path sharing.
func TestCompressedSources(t *testing.T) {
	dir := t.TempDir()
	plain, err := compileAlertTree(writeTiny(t))
	if err != nil {
		t.Fatal(err)
	}
	if got := alertBinPath(filepath.Join(dir, "last.json.zst")); got != filepath.Join(dir, "last.bin") {
		t.Errorf("zst bin path: %s", got)
	}
	if got := alertBinPath(filepath.Join(dir, "last.json.gz")); got != filepath.Join(dir, "last.bin") {
		t.Errorf("gz bin path: %s", got)
	}
	if got := alertBinPath(filepath.Join(dir, "last.json")); got != filepath.Join(dir, "last.bin") {
		t.Errorf("json bin path: %s", got)
	}

	var zbuf bytes.Buffer
	zw, err := zstd.NewWriter(&zbuf, zstd.WithEncoderLevel(zstd.SpeedBestCompression))
	if err != nil {
		t.Fatal(err)
	}
	zw.Write([]byte(tinyReport))
	zw.Close()
	zpath := filepath.Join(dir, "tree.json.zst")
	if err := os.WriteFile(zpath, zbuf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	var gbuf bytes.Buffer
	gw := gzip.NewWriter(&gbuf)
	gw.Write([]byte(tinyReport))
	gw.Close()
	gpath := filepath.Join(dir, "tree.json.gz")
	if err := os.WriteFile(gpath, gbuf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	for _, path := range []string{zpath, gpath} {
		got, err := compileAlertTree(path)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if len(got.nodes) != len(plain.nodes) || len(got.kids) != len(plain.kids) || got.deals != plain.deals {
			t.Fatalf("%s: tree shape differs", path)
		}
		for i := range plain.nodes {
			if got.nodes[i] != plain.nodes[i] {
				t.Fatalf("%s: node %d differs", path, i)
			}
		}
		// the compressed source also works through the cache path
		fi, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		bin := alertBinPath(path)
		if err := writeAlertBin(bin, got, fi); err != nil {
			t.Fatal(err)
		}
		if back, err := readAlertBin(bin, fi); err != nil || len(back.nodes) != len(plain.nodes) {
			t.Fatalf("%s: cache roundtrip failed: %v", path, err)
		}
	}
}

// TestRealTree compiles the shipped models/last.json[.zst|.gz] (skipped when
// absent) and spot-checks the head of the tree, then exercises the cache.
func TestRealTree(t *testing.T) {
	path := alertTreePath()
	fi, err := os.Stat(path)
	if err != nil {
		t.Skip("no real tree file:", err)
	}
	start := time.Now()
	tr, err := compileAlertTree(path)
	if err != nil {
		t.Fatal(err)
	}
	compile := time.Since(start)
	if tr.deals != 100000 {
		t.Errorf("deals = %d, want 100000", tr.deals)
	}
	// Values straight from the head of models/last.json:
	//   P: n=48254 hcp [0,18]; P,P: n=18703; P,P,P: n=4900; P,P,P,1C: n=1043 hcp [9,20]
	p := tr.walk([]string{"P"})
	if p < 0 || tr.nodes[p].n != 48254 || tr.nodes[p].hcpLo != 0 || tr.nodes[p].hcpHi != 18 {
		t.Fatalf("P node wrong: %d %+v", p, tr.nodes[p])
	}
	ppp := tr.walk([]string{"P", "P", "P"})
	if ppp < 0 || tr.nodes[ppp].n != 4900 {
		t.Fatalf("P,P,P node wrong: %d", ppp)
	}
	c := tr.walk([]string{"P", "P", "P", "1C"})
	if c < 0 || tr.nodes[c].n != 1043 || tr.nodes[c].hcpLo != 9 || tr.nodes[c].hcpHi != 20 {
		t.Fatalf("P,P,P,1C node wrong: %d", c)
	}
	if a := tr.alertAt(ppp, c); a == nil || a.N != 1043 {
		t.Fatalf("alert wrong: %+v", a)
	}
	// cache roundtrip on the real tree
	bin := filepath.Join(t.TempDir(), "last.bin")
	start = time.Now()
	if err := writeAlertBin(bin, tr, fi); err != nil {
		t.Fatal(err)
	}
	write := time.Since(start)
	start = time.Now()
	got, err := readAlertBin(bin, fi)
	if err != nil {
		t.Fatal(err)
	}
	read := time.Since(start)
	if len(got.nodes) != len(tr.nodes) || got.walk([]string{"P", "P", "P", "1C"}) != c {
		t.Fatal("cache roundtrip broke lookups")
	}
	// lookups must be microseconds
	start = time.Now()
	for i := 0; i < 10000; i++ {
		tr.walk([]string{"P", "1C", "P", "1N"})
		tr.optionAlerts([]string{"P", "1C"}, []string{"P", "1D", "1H", "1S", "1N", "X"})
	}
	perLookup := time.Since(start) / 20000
	t.Logf("nodes=%d kids=%d | compile=%s cacheWrite=%s cacheRead=%s lookup=%s",
		len(tr.nodes), len(tr.kids), compile.Round(time.Millisecond),
		write.Round(time.Millisecond), read.Round(time.Millisecond), perLookup)
	var ms runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms)
	t.Logf("live heap %d MiB, heap reserved from OS %d MiB", ms.HeapInuse>>20, ms.HeapSys>>20)
	bfi, _ := os.Stat(bin)
	t.Logf("cache size: %d KiB (source %d MiB)", bfi.Size()>>10, fi.Size()>>20)
}
