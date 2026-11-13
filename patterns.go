package main

// ---- Sequence-pattern layer of the system file (Go port) ----
//
// patterns.go is the Go twin of bidding_dt/conventions/patterns.py: it parses
// the sequence-pattern entries of models/system.txt, matches them against the
// live auction and renders their descriptions as the website's bid alerts.
// The grammar (generic suits X/Y/Z/W bound pairwise-distinct, opponent calls
// in parentheses, "..."-prefixed tail matching, OR-combined clause atoms,
// OR-split description branches, @weight/@! attributes) and every matching
// rule are shared with the Python side; tests/test_patterns.py (Python) and
// TestPatternGoldens (Go) pin the two implementations to the same cases.
// See models/system.txt's header for the full format documentation.
//
// The Go side only resolves and renders -- the hand-checking tokens
// (5+X, p>16, balanced, ...) and the force/double discipline are enforced by
// the trainer; here they are simply prettified into the alert text.

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// step kinds
const (
	stepBid = iota
	stepPass
	stepDbl
	stepRdbl
)

// atom kinds
const (
	atomCmp = iota
	atomCnt
	atomBidcmp
	atomGF
	atomUncontested
	atomContested
	atomBalancing
	atomPassed
	atomWeXX
	atomTheyXX
)

var (
	reStep    = regexp.MustCompile(`^([1-7])(NT|N|[CDHSXYZW])$`)
	reLen     = regexp.MustCompile(`^(\d+)(\+|-(\d+))?([CDHSNXYZW]|any)$`)
	rePts     = regexp.MustCompile(`^p(>=|<=|>|<)?(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?$`)
	reCmp     = regexp.MustCompile(`^([CDHSNXYZW])([<>])([CDHSNXYZW])$`)
	reCnt     = regexp.MustCompile(`^(dbls_by_us|dbls_by_them)>=(\d+)$`)
	reBidCmp  = regexp.MustCompile(`^(their_bid|our_bid)>=([1-7](?:NT|N|[CDHS]))$`)
	reIdent   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	reOrSplit = regexp.MustCompile(`\s+OR\s+`)
)

// callStepTokens are bare call names that may start a pattern line ("P = ..."
// describes a pass); they are NOT key = value lines even though identifiers.
var callStepTokens = map[string]bool{
	"P": true, "PASS": true, "X": true, "DBL": true, "D": true,
	"XX": true, "RDBL": true, "RD": true,
}

type patStep struct {
	them  bool // parenthesized: an opponent call
	kind  int
	level int
	denom byte // C D H S N or a variable letter X Y Z W
}

type patAtom struct {
	kind int
	a    byte // cmp: left ref; cnt/bidcmp: 'u's / 't'hem
	op   byte // cmp: '<' or '>'
	b    byte // cmp: right ref
	n    int  // cnt threshold
	call int  // bidcmp call id
}

type patToken struct {
	kind   int // tokLen | tokPts | tokBal | tokF | tokGF | tokTO | tokPenalty | tokText
	lo, hi float64
	op     string // pts: > < >= <= range exact
	suit   string // len: variable letter / literal / "any"
	text   string
}

// token kinds
const (
	tokLen = iota
	tokPts
	tokBal
	tokF
	tokGF
	tokTO
	tokPenalty
	tokText
)

type patEntry struct {
	raw        string
	pat        string
	steps      []patStep
	wildcard   bool
	atomGroups [][]patAtom
	concrete   map[int]bool // literal suit denoms of the pattern
	desc       string
	branches   [][]patToken
	flagF      bool
	flagGF     bool
	flagTO     bool
	flagPen    bool
	weight     float64
	artificial bool
}

func isVarChar(c byte) bool { return c == 'X' || c == 'Y' || c == 'Z' || c == 'W' }

func denomOfChar(c byte) int {
	switch c {
	case 'C':
		return 0
	case 'D':
		return 1
	case 'H':
		return 2
	case 'S':
		return 3
	case 'N':
		return denomN
	}
	return -1
}

// descSepIndex finds the description "=": the first one not part of a
// >= / <= clause comparison.
func descSepIndex(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == '=' && (i == 0 || (s[i-1] != '<' && s[i-1] != '>')) {
			return i
		}
	}
	return -1
}

// isKVLine reports whether a line's left-hand side makes it a key = value
// line (a bare identifier that is not a call token) rather than a pattern.
func isKVLine(lhs string) bool {
	return reIdent.MatchString(lhs) && !callStepTokens[strings.ToUpper(lhs)]
}

func parseStep(tok string) (patStep, error) {
	them := false
	tok = strings.TrimSpace(tok)
	if len(tok) >= 2 && strings.HasPrefix(tok, "(") && strings.HasSuffix(tok, ")") {
		them = true
		tok = strings.TrimSpace(tok[1 : len(tok)-1])
	}
	up := strings.ToUpper(tok)
	switch up {
	case "P", "PASS":
		return patStep{them: them, kind: stepPass}, nil
	case "X", "DBL", "D":
		return patStep{them: them, kind: stepDbl}, nil
	case "XX", "RDBL", "RD":
		return patStep{them: them, kind: stepRdbl}, nil
	}
	m := reStep.FindStringSubmatch(up)
	if m == nil {
		return patStep{}, fmt.Errorf("bad pattern step %q", tok)
	}
	lvl, _ := strconv.Atoi(m[1])
	d := m[2]
	if d == "NT" {
		d = "N"
	}
	return patStep{them: them, kind: stepBid, level: lvl, denom: d[0]}, nil
}

func parseAtom(a string) (patAtom, error) {
	if m := reCmp.FindStringSubmatch(a); m != nil {
		return patAtom{kind: atomCmp, a: m[1][0], op: m[2][0], b: m[3][0]}, nil
	}
	if m := reCnt.FindStringSubmatch(a); m != nil {
		n, _ := strconv.Atoi(m[2])
		which := byte('u')
		if m[1] == "dbls_by_them" {
			which = 't'
		}
		return patAtom{kind: atomCnt, a: which, n: n}, nil
	}
	if m := reBidCmp.FindStringSubmatch(a); m != nil {
		id, ok := vocabID(m[2])
		if !ok {
			return patAtom{}, fmt.Errorf("bad call in atom %q", a)
		}
		which := byte('t')
		if m[1] == "our_bid" {
			which = 'u'
		}
		return patAtom{kind: atomBidcmp, a: which, call: int(id)}, nil
	}
	switch a {
	case "gf":
		return patAtom{kind: atomGF}, nil
	case "uncontested":
		return patAtom{kind: atomUncontested}, nil
	case "contested":
		return patAtom{kind: atomContested}, nil
	case "balancing":
		return patAtom{kind: atomBalancing}, nil
	case "passed":
		return patAtom{kind: atomPassed}, nil
	case "we_xx":
		return patAtom{kind: atomWeXX}, nil
	case "they_xx":
		return patAtom{kind: atomTheyXX}, nil
	}
	return patAtom{}, fmt.Errorf("bad clause atom %q", a)
}

func parseAtoms(clause string) ([][]patAtom, error) {
	if strings.TrimSpace(clause) == "" {
		return nil, nil
	}
	var groups [][]patAtom
	for _, g := range reOrSplit.Split(clause, -1) {
		var atoms []patAtom
		for _, a := range strings.FieldsFunc(g, func(r rune) bool {
			return r == ',' || r == ' ' || r == '\t'
		}) {
			atom, err := parseAtom(a)
			if err != nil {
				return nil, err
			}
			atoms = append(atoms, atom)
		}
		if len(atoms) > 0 {
			groups = append(groups, atoms)
		}
	}
	if len(groups) == 0 {
		return nil, fmt.Errorf("empty clause %q", clause)
	}
	return groups, nil
}

func parseToken(w string) patToken {
	if m := reLen.FindStringSubmatch(w); m != nil {
		n, _ := strconv.ParseFloat(m[1], 64)
		lo, hi := n, n
		if m[2] == "+" {
			hi = 99
		} else if strings.HasPrefix(m[2], "-") {
			hi, _ = strconv.ParseFloat(m[3], 64)
		}
		return patToken{kind: tokLen, lo: lo, hi: hi, suit: m[4]}
	}
	if m := rePts.FindStringSubmatch(w); m != nil {
		v, _ := strconv.ParseFloat(m[2], 64)
		if m[3] != "" {
			hi, _ := strconv.ParseFloat(m[3], 64)
			return patToken{kind: tokPts, lo: v, hi: hi, op: "range"}
		}
		op := m[1]
		if op == "" {
			op = "exact"
		}
		return patToken{kind: tokPts, lo: v, hi: v, op: op}
	}
	switch w {
	case "F":
		return patToken{kind: tokF}
	case "GF":
		return patToken{kind: tokGF}
	case "TO":
		return patToken{kind: tokTO}
	case "PENALTY":
		return patToken{kind: tokPenalty}
	case "balanced", "bal":
		return patToken{kind: tokBal}
	}
	return patToken{kind: tokText, text: w}
}

// parseEntryLine parses one pattern line of the system file.
func parseEntryLine(line string) (*patEntry, error) {
	raw := strings.TrimSpace(line)
	lhs, desc := raw, ""
	if i := descSepIndex(raw); i >= 0 {
		lhs, desc = strings.TrimSpace(raw[:i]), strings.TrimSpace(raw[i+1:])
	}
	pat, clause := lhs, ""
	if i := strings.IndexAny(lhs, ";:"); i >= 0 {
		pat, clause = strings.TrimSpace(lhs[:i]), strings.TrimSpace(lhs[i+1:])
	}
	toks := make([]string, 0, 4)
	for _, t := range strings.Split(pat, "-") {
		if strings.TrimSpace(t) != "" {
			toks = append(toks, strings.TrimSpace(t))
		}
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("empty pattern: %s", raw)
	}
	wildcard := false
	if toks[0] == "..." {
		wildcard = true
		toks = toks[1:]
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("wildcard without steps: %s", raw)
	}
	e := &patEntry{raw: raw, pat: pat, wildcard: wildcard, desc: desc,
		concrete: map[int]bool{}, weight: 1.0}
	for _, t := range toks {
		s, err := parseStep(t)
		if err != nil {
			return nil, err
		}
		e.steps = append(e.steps, s)
		if s.kind == stepBid {
			if d := denomOfChar(s.denom); d >= 0 && d < denomN {
				e.concrete[d] = true
			}
		}
	}
	groups, err := parseAtoms(clause)
	if err != nil {
		return nil, err
	}
	e.atomGroups = groups

	var cur []patToken
	for _, w := range strings.Fields(desc) {
		if strings.HasPrefix(w, "@") {
			attr := w[1:]
			if attr == "!" {
				e.artificial = true
				continue
			}
			f, err := strconv.ParseFloat(attr, 64)
			if err != nil {
				return nil, fmt.Errorf("bad attribute %q in %q", w, raw)
			}
			e.weight = f
			continue
		}
		if w == "OR" {
			e.branches = append(e.branches, cur)
			cur = nil
			continue
		}
		tk := parseToken(w)
		switch tk.kind {
		case tokF:
			e.flagF = true
		case tokGF:
			e.flagGF = true
		case tokTO:
			e.flagTO = true
		case tokPenalty:
			e.flagPen = true
		}
		cur = append(cur, tk)
	}
	e.branches = append(e.branches, cur)
	if e.flagTO && e.flagPen {
		return nil, fmt.Errorf("entry is both TO and PENALTY: %s", raw)
	}
	return e, nil
}

// ---- Matching ----

func callMatchesStep(step patStep, call int, binds map[byte]int,
	concrete map[int]bool) bool {
	switch step.kind {
	case stepPass:
		return call == int(callPass)
	case stepDbl:
		return call == int(callX)
	case stepRdbl:
		return call == int(callXX)
	}
	if call < firstBid {
		return false
	}
	level := (call-firstBid)/5 + 1
	denom := (call - firstBid) % 5
	if level != step.level {
		return false
	}
	d := step.denom
	if d == 'N' {
		return denom == denomN
	}
	if isVarChar(d) {
		if denom == denomN {
			return false // variables are suits, never NT
		}
		if prev, ok := binds[d]; ok {
			return prev == denom
		}
		if concrete[denom] {
			return false // distinct from the pattern's literal suits
		}
		for _, v := range binds {
			if v == denom {
				return false // variables bind pairwise-distinct suits
			}
		}
		binds[d] = denom
		return true
	}
	return denom == denomOfChar(d)
}

// matchFrom binds e.steps onto seq[start:], consuming exactly through the
// candidate (the last seq element); ok is false when no alignment works.
func matchFrom(e *patEntry, dealer int, seq []int, us, start int) (map[byte]int, bool) {
	binds := map[byte]int{}
	i := start
	for _, step := range e.steps {
		if !step.them && step.kind != stepPass {
			for i < len(seq) { // implicit opponent-pass skip
				seat := (dealer + i) % 4
				if seat%2 != us && seq[i] == int(callPass) {
					i++
					continue
				}
				break
			}
		}
		if i >= len(seq) {
			return nil, false
		}
		seat := (dealer + i) % 4
		if step.them != (seat%2 != us) {
			return nil, false
		}
		if !callMatchesStep(step, seq[i], binds, e.concrete) {
			return nil, false
		}
		i++
	}
	return binds, i == len(seq)
}

func denomRef(c byte, binds map[byte]int) int {
	if isVarChar(c) {
		if v, ok := binds[c]; ok {
			return v
		}
		return -1
	}
	return denomOfChar(c)
}

// matchState is the auction state the clause conditions read, built by
// replaying the calls from the dealer (the Go twin of patterns.MatchState,
// limited to what matching needs: the discipline fields are training-only).
type matchState struct {
	gf             [2]bool // file GF latches OR the structural game force
	gfFile         [2]bool
	dbls           [2]int
	xx             [2]bool
	high           [2]int
	activity       [2]int
	passesSinceBid int
	called         [4]bool
	bidSeat        [4]bool
	reader         *auctionReader // parallel reader (structural gf)
}

func newMatchState(dealer int) *matchState {
	return &matchState{reader: newAuctionReader(dealer)}
}

// foldState folds call #i (dealer order) into st; the entry match for the
// call uses the state BEFORE it, exactly like the Python replay.
func foldState(entries []*patEntry, dealer int, st *matchState, i, call int) {
	seat := (dealer + i) % 4
	side := seat % 2
	var entry *patEntry
	if len(entries) > 0 {
		entry, _ = matchFirst(entries, dealer, st.reader.calls, call, st)
	}
	st.called[seat] = true
	live := entry != nil && entry.weight > 0
	switch call {
	case int(callPass):
		st.passesSinceBid++
	case int(callX):
		st.dbls[side]++
		st.passesSinceBid = 0
		st.activity[side]++
	case int(callXX):
		st.xx[side] = true
		st.passesSinceBid = 0
		st.activity[side]++
	default:
		denom := (call - firstBid) % 5
		level := (call-firstBid)/5 + 1
		if call > st.high[side] {
			st.high[side] = call
		}
		st.passesSinceBid = 0
		st.activity[side]++
		st.bidSeat[seat] = true
		if live && entry.flagGF {
			st.gfFile[side] = true
		}
		if isGameOrAbove(level, denom) {
			st.gfFile[side] = false // obligation fulfilled
		}
	}
	st.reader.feed(call)
	st.gf = [2]bool{st.gfFile[0] || st.reader.gf[0],
		st.gfFile[1] || st.reader.gf[1]}
}

func atomsOk(groups [][]patAtom, binds map[byte]int, st *matchState,
	us, seat int) bool {
	if len(groups) == 0 {
		return true
	}
	for _, g := range groups { // OR over groups
		ok := true
		for _, a := range g { // AND within a group
			switch a.kind {
			case atomCmp:
				x, y := denomRef(a.a, binds), denomRef(a.b, binds)
				if x < 0 || y < 0 {
					ok = false
				} else if a.op == '<' {
					ok = x < y
				} else {
					ok = x > y
				}
			case atomGF:
				ok = st.gf[us]
			case atomUncontested:
				ok = st.activity[1-us] == 0
			case atomContested:
				ok = st.activity[1-us] > 0
			case atomBalancing:
				ok = st.passesSinceBid >= 2
			case atomPassed:
				ok = st.called[seat] && !st.bidSeat[seat]
			case atomWeXX:
				ok = st.xx[us]
			case atomTheyXX:
				ok = st.xx[1-us]
			case atomCnt:
				n := st.dbls[us]
				if a.a == 't' {
					n = st.dbls[1-us]
				}
				ok = n >= a.n
			case atomBidcmp:
				h := st.high[1-us]
				if a.a == 'u' {
					h = st.high[us]
				}
				ok = h >= a.call
			}
			if !ok {
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// matchFirst returns the first entry (file order) matching `action` after
// `calls`, with its suit-variable bindings; (nil, nil) when none matches.
func matchFirst(entries []*patEntry, dealer int, calls []int, action int,
	st *matchState) (*patEntry, map[byte]int) {
	seq := make([]int, 0, len(calls)+1)
	seq = append(seq, calls...)
	seq = append(seq, action)
	seat := (dealer + len(seq) - 1) % 4
	us := seat % 2
	for _, e := range entries {
		var starts []int
		switch {
		case e.wildcard:
			for s := 0; s < len(seq); s++ {
				starts = append(starts, s)
			}
		case e.steps[0].kind == stepPass:
			starts = []int{0}
		default:
			s0 := -1
			for i, c := range seq {
				if c != int(callPass) {
					s0 = i
					break
				}
			}
			if s0 >= 0 {
				starts = []int{s0}
			}
		}
		for _, s := range starts {
			if binds, ok := matchFrom(e, dealer, seq, us, s); ok &&
				atomsOk(e.atomGroups, binds, st, us, seat) {
				return e, binds
			}
		}
	}
	return nil, nil
}

// ---- Alert rendering (mirrors patterns.render_desc) ----

func suitSym(ref string, binds map[byte]int) string {
	if ref == "any" {
		return "any"
	}
	d := -1
	if isVarChar(ref[0]) {
		if v, ok := binds[ref[0]]; ok {
			d = v
		}
	} else {
		d = denomOfChar(ref[0])
	}
	if d >= 0 && d < denomN {
		return suitSymbols[d]
	}
	return "any"
}

func prettyToken(w string, binds map[byte]int) string {
	if m := reLen.FindStringSubmatch(w); m != nil {
		sym := suitSym(m[4], binds)
		if m[2] == "+" {
			return m[1] + "+ " + sym
		}
		if strings.HasPrefix(m[2], "-") {
			return m[1] + "-" + m[3] + " " + sym
		}
		return m[1] + " " + sym
	}
	if m := rePts.FindStringSubmatch(w); m != nil {
		if m[3] != "" {
			return m[2] + "-" + m[3] + " pts"
		}
		switch m[1] {
		case ">", ">=":
			return m[2] + "+ pts"
		case "<", "<=":
			return "under " + m[2] + " pts"
		}
		return m[2] + " pts"
	}
	switch w {
	case "F":
		return "forcing"
	case "GF":
		return "game force"
	case "TO":
		return "takeout"
	case "PENALTY":
		return "penalty"
	case "OR":
		return "or"
	}
	return w
}

// renderDesc is the alert text of an entry: suit variables resolved and the
// enforcement tokens prettified; @attributes dropped.
func renderDesc(e *patEntry, binds map[byte]int) string {
	words := make([]string, 0, 8)
	for _, w := range strings.Fields(e.desc) {
		if strings.HasPrefix(w, "@") {
			continue
		}
		words = append(words, prettyToken(w, binds))
	}
	return strings.Join(words, " ")
}
