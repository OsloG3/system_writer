package main

// ---- The bot's system file (models/system.txt) ----
//
// system.txt is the conventions text file the bidding bot is trained from:
// bidding-dt interprets it at the start of training (rule weights + numeric
// thresholds for the convention checker, see bidding_dt/conventions/
// system_file.py) and this server generates the website's bid alerts from it
// instead of any self-play statistics: the auction reader (auction.go)
// classifies each bot call, sectionFor resolves the call to the one system
// section whose agreement it is, and that section's alert template is
// rendered with the call's suits. The section resolution mirrors the Python
// SystemSpec.section_for exactly -- keep the two in lockstep.
//
// Format (INI-like, # and ; comments):
//
//	[section]           call kind; names match the Python rule codes plus the
//	                    alert-only kinds (rkcb_ask, resp_2c_waiting,
//	                    balancing_takeout, penalty_double)
//	alert = text        what the website shows; placeholders {call} {suit}
//	                    {other} {trump} {partner_suit} {shows} {level}
//	artificial = yes    show the call with the alert (!) marker
//	steps = a | b | c | d   RKCB step-response texts ({shows})
//
// Any other keys (weights, thresholds) belong to the training side and are
// ignored here. The file is small and re-read whenever its mtime changes, so
// editing it refreshes the alerts without a restart. SYSTEM_FILE overrides
// the path.

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

// systemSection is one parsed [section] of the system file.
type systemSection struct {
	alert      string
	artificial bool
	steps      []string
}

// systemSpec is the parsed system file.
type systemSpec struct {
	sections map[string]*systemSection
	nAlerts  int // sections carrying alert text
}

// systemFilePath resolves the system file: SYSTEM_FILE, else the default.
func systemFilePath() string {
	if p := strings.TrimSpace(os.Getenv("SYSTEM_FILE")); p != "" {
		return p
	}
	return defaultSystemFile
}

// parseSystem turns system-file text into a spec. Malformed structural lines
// fail loudly; unknown keys are the training side's business and skipped.
func parseSystem(text string) (*systemSpec, error) {
	spec := &systemSpec{sections: map[string]*systemSection{}}
	var cur *systemSection
	var curName string
	for i, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(strings.TrimSuffix(raw, "\r"))
		if line == "" || line[0] == '#' || line[0] == ';' {
			continue
		}
		where := fmt.Sprintf("line %d", i+1)
		if strings.HasPrefix(line, "[") {
			if !strings.HasSuffix(line, "]") {
				return nil, fmt.Errorf("%s: unterminated section header", where)
			}
			curName = strings.TrimSpace(line[1 : len(line)-1])
			if curName == "" {
				return nil, fmt.Errorf("%s: empty section name", where)
			}
			if _, dup := spec.sections[curName]; dup {
				return nil, fmt.Errorf("%s: duplicate section [%s]", where, curName)
			}
			cur = &systemSection{}
			spec.sections[curName] = cur
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("%s: expected key = value", where)
		}
		if cur == nil {
			return nil, fmt.Errorf("%s: key outside any section", where)
		}
		key, val = strings.TrimSpace(key), strings.TrimSpace(val)
		switch key {
		case "alert":
			cur.alert = val
			spec.nAlerts++
		case "artificial":
			v := strings.ToLower(val)
			cur.artificial = v == "yes" || v == "true"
		case "steps":
			for _, s := range strings.Split(val, "|") {
				cur.steps = append(cur.steps, strings.TrimSpace(s))
			}
		}
	}
	if len(spec.sections) == 0 {
		return nil, errors.New("no sections found")
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

// ---- Call -> section resolution ----
//
// Mirrors bidding_dt.conventions.system_file.section_for, which mirrors the
// check order of rules._check_bid: the same call resolves to the same section
// on both sides of the wire.

// dblSections maps the double contexts onto their system sections.
var dblSections = map[string]string{
	"takeout":           "takeout_double",
	"balancing_takeout": "balancing_takeout",
	"negative":          "negative_double",
	"penalty":           "penalty_double",
	"nt_penalty":        "penalty_double",
}

func sectionFor(info callInfo) string {
	switch info.kind {
	case kindPass:
		if info.xyzAfterRelay {
			return "xyz_pass"
		}
		return ""
	case kindDouble:
		return dblSections[info.dblCtx]
	case kindRedouble:
		return ""
	}

	if info.opening {
		if info.denom == denomN {
			switch info.level {
			case 1:
				return "open_1nt_range"
			case 2:
				return "open_2nt_range"
			}
			return ""
		}
		if info.level == 2 {
			if info.denom == 0 {
				return "open_2c_strong"
			}
			return "open_weak2"
		}
		if info.level == 1 {
			if isMajor(info.denom) {
				return "open_major_5card"
			}
			return "open_minor_length"
		}
		return "" // 3+ preempts: undefined
	}
	if info.sideStrong2c {
		if info.strong2c && info.firstCall {
			if info.denom == 1 && info.level == 2 {
				return "resp_2c_waiting"
			}
			if info.denom < denomN || info.level == 2 {
				return "resp_2c_positive"
			}
		}
		return ""
	}
	switch {
	case info.xyz2c:
		return "xyz_2c"
	case info.xyz2d:
		return "xyz_2d"
	case info.xyzRelay:
		return "xyz_relay"
	case info.smolen:
		return "smolen"
	case info.rkcbAsk:
		return "rkcb_ask"
	case info.rkcbResponse:
		return "rkcb_response"
	case info.is2over1:
		return "two_one_gf"
	case info.fourthSuit:
		return "fourth_suit"
	case info.isJumpShift:
		return "inv_jump_shift"
	case info.isStrongJS:
		return "strong_jump_rebid"
	case info.splinter:
		return "splinter"
	}

	if info.denom == denomN {
		if info.level == 4 {
			return "" // quantitative 4NT: undefined
		}
		if info.overcall && info.level == 1 {
			if info.balancing {
				return "overcall_1nt_bal"
			}
			return "overcall_1nt"
		}
		if info.responder || info.advancer {
			switch {
			case info.level == 1:
				return "resp_1nt"
			case info.level == 2 && info.jump == 1:
				return "resp_2nt"
			case info.level == 3 && info.jump == 2 && info.responder && info.firstCall:
				return "resp_3nt"
			}
			return ""
		}
		if info.openerRebid {
			if info.level == 1 {
				return "rebid_1nt"
			}
			if info.level == 2 && info.jump == 1 {
				return "rebid_2nt"
			}
		}
		return ""
	}

	if info.cuebid {
		if info.partnerLastSuit >= 0 {
			return "cuebid_support"
		}
		return ""
	}
	if info.ownRebid {
		return ""
	}
	if info.raiseBid {
		if info.firstCall && !isGameOrAbove(info.level, info.denom) {
			switch info.jump {
			case 0:
				return "resp_raise"
			case 1:
				return "resp_jump_raise"
			}
		}
		return ""
	}
	if info.overcall && info.newSuit && info.jump == 0 {
		return "overcall"
	}
	return ""
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

// ---- Alert rendering ----

var suitSymbols = [4]string{"♣", "♦", "♥", "♠"} // project denom order C,D,H,S

func denomSymbol(denom int) string {
	if denom >= 0 && denom < denomN {
		return suitSymbols[denom]
	}
	return "NT"
}

// prettyCall renders the call the way the alert texts refer to it.
func prettyCall(info callInfo) string {
	switch info.kind {
	case kindPass:
		return "Pass"
	case kindDouble:
		return "Double"
	case kindRedouble:
		return "Redouble"
	}
	return fmt.Sprintf("%d%s", info.level, denomSymbol(info.denom))
}

// renderAlert fills the placeholders of one section's alert template.
func renderAlert(sec *systemSection, info callInfo) string {
	out := sec.alert
	if strings.Contains(out, "{") {
		out = strings.ReplaceAll(out, "{call}", prettyCall(info))
		out = strings.ReplaceAll(out, "{level}", fmt.Sprint(info.level))
		out = strings.ReplaceAll(out, "{suit}", denomSymbol(info.denom))
		if strings.Contains(out, "{other}") {
			other := 2 // H
			if info.denom == 2 {
				other = 3 // S
			}
			out = strings.ReplaceAll(out, "{other}", denomSymbol(other))
		}
		out = strings.ReplaceAll(out, "{trump}", denomSymbol(info.rkcbTrump))
		out = strings.ReplaceAll(out, "{partner_suit}", denomSymbol(info.partnerLastSuit))
		if strings.Contains(out, "{shows}") {
			shows := "an unspecified keycard count"
			if info.rkcbStep >= 1 && info.rkcbStep <= len(sec.steps) {
				shows = sec.steps[info.rkcbStep-1]
			}
			out = strings.ReplaceAll(out, "{shows}", shows)
		}
	}
	return out
}

// alertFor renders the client-visible alert for a classified call, or nil
// when the system file has no meaning for it.
func (s *systemSpec) alertFor(info callInfo) *bidAlert {
	name := sectionFor(info)
	if name == "" {
		return nil
	}
	sec := s.sections[name]
	if sec == nil || sec.alert == "" {
		return nil
	}
	return &bidAlert{Text: renderAlert(sec, info), Artificial: sec.artificial}
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
	log.Printf("Bid alerts ready: %d sections (%d with alert text) from %s",
		len(spec.sections), spec.nAlerts, sysPath)
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
