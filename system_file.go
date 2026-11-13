package main

// ---- The bot's system file (models/system.txt) ----
//
// system.txt is the conventions text file the bidding bot is trained from:
// bidding-dt interprets it at the start of training (see
// bidding_dt/conventions/system_file.py + patterns.py) and this server
// generates the website's bid alerts from it instead of any self-play
// statistics: the auction reader (auction.go) plus the pattern matcher
// (patterns.go) resolve each bot call to the first matching pattern entry of
// the file and serve its rendered description.
//
// The file is a sequence of pattern entries, one per line:
//
//	PATTERN[; CLAUSES] = DESCRIPTION [@weight] [@!]
//
// plus top-level `key = value` config lines (bare identifiers) that tune the
// trainer -- the Go side accepts and ignores them. Lines starting with # or ;
// are comments. INI-style [sections] are gone; a header line is an error.
// The grammar is documented in the file's own header and mirrored 1:1 by the
// Python parser. The file is small and re-read whenever its mtime changes,
// so editing it refreshes the alerts without a restart. SYSTEM_FILE
// overrides the path.

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"
)

const defaultSystemFile = "models/system.txt" // SYSTEM_FILE overrides

// systemSpec is the parsed system file.
type systemSpec struct {
	entries []*patEntry // pattern entries, in file order
	nAlerts int         // entries carrying alert text
}

// systemFilePath resolves the system file: SYSTEM_FILE, else the default.
func systemFilePath() string {
	if p := strings.TrimSpace(os.Getenv("SYSTEM_FILE")); p != "" {
		return p
	}
	return defaultSystemFile
}

// parseSystem turns system-file text into a spec. Malformed lines fail
// loudly; config lines (bare identifier = value) belong to the training side
// and are accepted and ignored here.
func parseSystem(text string) (*systemSpec, error) {
	spec := &systemSpec{}
	for i, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(strings.TrimSuffix(raw, "\r"))
		if line == "" || line[0] == '#' || line[0] == ';' {
			continue
		}
		where := fmt.Sprintf("line %d", i+1)
		if strings.HasPrefix(line, "[") {
			return nil, fmt.Errorf("%s: INI sections are gone -- write "+
				"pattern entries and top-level key = value config", where)
		}
		lhs := line
		if idx := descSepIndex(line); idx >= 0 {
			lhs = strings.TrimSpace(line[:idx])
		}
		if isKVLine(lhs) {
			if descSepIndex(line) < 0 {
				return nil, fmt.Errorf("%s: expected key = value", where)
			}
			continue // trainer config: accepted, nothing to render from it
		}
		e, err := parseEntryLine(line)
		if err != nil {
			return nil, fmt.Errorf("%s: %v", where, err)
		}
		if e.desc != "" {
			spec.nAlerts++
		}
		spec.entries = append(spec.entries, e)
	}
	if len(spec.entries) == 0 {
		return nil, errors.New("no pattern entries found")
	}
	return spec, nil
}

// loadSystemFile reads and parses the system file at path.
func loadSystemFile(path string) (*systemSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseSystem(string(data))
}

// isGameOrAbove: 3NT+, 4M+, 5m+ (and any 6/7-level bid) fulfill a game force.
func isGameOrAbove(level, denom int) bool {
	if level >= 6 {
		return true
	}
	if denom == denomN {
		return level >= 3
	}
	if isMajor(denom) {
		return level >= 4
	}
	return level >= 5
}

var suitSymbols = [4]string{"♣", "♦", "♥", "♠"} // project denom order C,D,H,S

// alertForCall renders the client-visible alert for one candidate call in
// the auction the match state was built from: the first matching pattern
// entry's rendered description, or nil when no entry matches. An entry whose
// rendered description is empty suppresses the alert.
func (s *systemSpec) alertForCall(st *matchState, call int) *bidAlert {
	e, binds := matchFirst(s.entries, st.reader.dealer, st.reader.calls,
		call, st)
	if e == nil {
		return nil
	}
	if text := renderDesc(e, binds); text != "" {
		return &bidAlert{Text: text, Artificial: e.artificial}
	}
	return nil
}

// ---- Loading (with mtime-based refresh) ----

// sysPath is the resolved system-file location; tests may point it elsewhere.
var sysPath = systemFilePath()

// sysState guards the cached spec and the source stamp it was built from.
var sysState struct {
	mu   sync.Mutex
	spec *systemSpec
	mod  time.Time
	size int64
}

// systemReady returns the parsed system file, (re)loading it when the file on
// disk is newer than the cached spec. Nil while the file is missing or
// unparsable -- boards are then simply served without alerts.
func systemReady() *systemSpec {
	fi, err := os.Stat(sysPath)
	if err != nil {
		return nil
	}
	sysState.mu.Lock()
	defer sysState.mu.Unlock()
	if sysState.spec != nil && fi.ModTime().Equal(sysState.mod) && fi.Size() == sysState.size {
		return sysState.spec
	}
	spec, err := loadSystemFile(sysPath)
	if err != nil {
		sysState.spec, sysState.mod, sysState.size = nil, time.Time{}, 0
		log.Printf("Bid alerts unavailable (%s): %v", sysPath, err)
		return nil
	}
	sysState.spec, sysState.mod, sysState.size = spec, fi.ModTime(), fi.Size()
	log.Printf("Bid alerts ready: %d pattern entries (%d with alert text) from %s",
		len(spec.entries), spec.nAlerts, sysPath)
	return spec
}

// initAlerts resolves the system-file path and primes the cache.
func initAlerts() {
	sysPath = systemFilePath()
	if systemReady() == nil {
		if _, err := os.Stat(sysPath); errors.Is(err, os.ErrNotExist) {
			log.Printf("Bid alerts off: no system file at %s (set SYSTEM_FILE)", sysPath)
		}
	}
}
