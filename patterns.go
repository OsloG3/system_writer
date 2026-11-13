package main

// ---- Sequence-pattern layer of the system file (Go port) ----
//
// patterns.go is the Go twin of bidding_dt/conventions/patterns.py: it parses
// the pattern entries of models/system.txt, matches them against the live
// auction and renders their descriptions as the website's bid alerts. The
// grammar (generic suits X/Y/Z/W bound pairwise-distinct, opponent calls in
// parentheses, "..." wildcard steps, "?" any-level, "!"-negated steps,
// OR-combined clause atoms over suit comparisons / candidate shape / auction
// state, OR-split description branches, @weight/@! attributes) and every
// matching rule are shared with the Python side; tests/test_patterns.py
// (Python) and patterns_test.go (Go) pin the two implementations to the same
// cases. See models/system.txt's header for the full format documentation.
//
// The Go side only resolves and renders -- the hand-checking tokens
// (5+X, p>16, balanced, kc=1/4, ...) and the force/double discipline are
// enforced by the trainer; here they are simply prettified into the alert
// text.

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
	stepAny // "..." -- absorbs any run of calls
)

// atom kinds
const (
	atomCmp = iota
	atomCnt
	atomNum
	atomBidcmp
	atomBelow
	atomCond
)

// condition words (atomCond)
const (
	condGF = iota
	condUncontested
	condContested
	condBalancing
	condPassed
	condFirst
	condOpener
	condPartnerbid
	condNewsuit
	condWebid
	condNotopening
	condNoxyz
	condNostrong2c
	condFourth
	condWeXX
	condTheyXX
)

var condWords = map[string]int{
	"gf": condGF, "uncontested": condUncontested, "contested": condContested,
	"balancing": condBalancing, "passed": condPassed, "first": condFirst,
	"opener": condOpener, "partnerbid": condPartnerbid,
	"newsuit": condNewsuit, "we_bid": condWebid,
	"notopening": condNotopening, "noxyz": condNoxyz,
	"nostrong2c": condNostrong2c, "fourth": condFourth,
	"we_xx": condWeXX, "they_xx": condTheyXX,
}

var (
	reStep    = regexp.MustCompile(`^([1-7?])(NT|N|[CDHSXYZW])$`)
	reLen     = regexp.MustCompile(`^(\d+)(\+|-(\d+))?([CDHSNXYZWTPLO]|any)$`)
	rePts     = regexp.MustCompile(`^p(>=|<=|>|<)?(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?$`)
	reKC      = regexp.MustCompile(`^kc=(0/3|1/4|2-|2\+)$`)
	reCmp     = regexp.MustCompile(`^([CDHSNXYZWTPL])(>=|<=|!=|[<>]|=)([CDHSNXYZWTPL])$`)
	reCnt     = regexp.MustCompile(`^(dbls|bids)_by_(us|them)(>=|=)(\d+)$`)
	reNum     = regexp.MustCompile(`^(jump|level)(>=|<=|=)(\d+)$`)
	reBidCmp  = regexp.MustCompile(`^(their_bid|our_bid|below)(>=|=)([1-7](?:NT|N|[CDHS]))$`)
	reIdent   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	reOrSplit = regexp.MustCompile(`\s+OR\s+`)
	reBrace   = regexp.MustCompile(`\{([XYZWTPLO])\}`)
)

// callStepTokens are bare call names that may start a pattern line ("P = ..."
// describes a pass); they are NOT config lines even though identifiers.
var callStepTokens = map[string]bool{
	"P": true, "PASS": true, "X": true, "DBL": true, "D": true,
	"XX": true, "RDBL": true, "RD": true,
}

type patStep struct {
	them    bool // parenthesized: an opponent call
	kind    int
	level   int  // 0 = any level ("?Y")
	denom   byte // C D H S N or a variable letter X Y Z W
	negated bool // "!2D": any of our bids except this one
}

type patAtom struct {
	kind    int
	cond    int    // atomCond: which condition
	a       byte   // cmp: left ref; cnt: 'u's/'t'hem + which counter; bidcmp: which
	cntWhat byte   // cnt: 'd'oubles / 'b'ids
	op      string // cmp/num/cnt: < > >= <= = !=
	b       byte   // cmp: right ref
	n       int    // cnt/num threshold
	call    int    // bidcmp/below call id
	negated bool
}

type patTokenKind int

// token kinds
const (
	tokText patTokenKind = iota
	tokLen
	tokPts
	tokKC
	tokKeyword
)

type patToken struct {
	kind patTokenKind
	// (rendering only -- the Go side never evaluates tokens)
	text string
}

type patEntry struct {
	raw        string
	pat        string
	steps      []patStep
	anchored   bool // no leading "..." step
	atomGroups [][]patAtom
	desc       string
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

// descSepIndex finds the description separator: an "=" surrounded by
// whitespace (or ending the line). Clause atoms keep their tight "=" / ">="
// (jump=0, bids_by_them=0, their_bid>=4S) precisely because those have no
// space before the "=".
func descSepIndex(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] != '=' || i == 0 {
			continue
		}
		if !isSpaceByte(s[i-1]) {
			continue
		}
		if i+1 == len(s) || isSpaceByte(s[i+1]) {
			return i
		}
	}
	return -1
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

// isKVLine reports whether a line's left-hand side makes it a key = value
// config line (a bare identifier that is not a call token) rather than a
// pattern entry.
func isKVLine(lhs string) bool {
	return reIdent.MatchString(lhs) && !callStepTokens[strings.ToUpper(lhs)]
}

func parseStep(tok string) (patStep, error) {
	them := false
	tok = strings.TrimSpace(tok)
	if tok == "..." {
		return patStep{kind: stepAny}, nil
	}
	if len(tok) >= 2 && strings.HasPrefix(tok, "(") && strings.HasSuffix(tok, ")") {
		them = true
		tok = strings.TrimSpace(tok[1 : len(tok)-1])
	}
	negated := false
	if strings.HasPrefix(tok, "!") {
		negated = true
		tok = strings.TrimSpace(tok[1:])
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
	level := 0 // "?" = any level
	if m[1] != "?" {
		level, _ = strconv.Atoi(m[1])
	}
	d := m[2]
	if d == "NT" {
		d = "N"
	}
	return patStep{them: them, kind: stepBid, level: level, denom: d[0],
		negated: negated}, nil
}

func parseAtom(a string) (patAtom, error) {
	negated := false
	if strings.HasPrefix(a, "!") {
		negated = true
		a = a[1:]
	}
	if m := reCmp.FindStringSubmatch(a); m != nil {
		return patAtom{kind: atomCmp, a: m[1][0], op: m[2], b: m[3][0],
			negated: negated}, nil
	}
	if m := reCnt.FindStringSubmatch(a); m != nil {
		n, _ := strconv.Atoi(m[4])
		what := byte('d')
		if m[1] == "bids" {
			what = 'b'
		}
		side := byte('u')
		if m[2] == "them" {
			side = 't'
		}
		return patAtom{kind: atomCnt, cntWhat: what, a: side, op: m[3],
			n: n, negated: negated}, nil
	}
	if m := reNum.FindStringSubmatch(a); m != nil {
		n, _ := strconv.Atoi(m[3])
		what := byte('j')
		if m[1] == "level" {
			what = 'l'
		}
		return patAtom{kind: atomNum, cntWhat: what, op: m[2], n: n,
			negated: negated}, nil
	}
	if m := reBidCmp.FindStringSubmatch(a); m != nil {
		id, ok := vocabID(strings.Replace(m[3], "NT", "N", 1))
		if !ok {
			return patAtom{}, fmt.Errorf("bad call in atom %q", a)
		}
		if m[1] == "below" {
			if m[2] != "=" {
				return patAtom{}, fmt.Errorf("bad clause atom %q", a)
			}
			return patAtom{kind: atomBelow, call: int(id),
				negated: negated}, nil
		}
		if m[2] != ">=" {
			return patAtom{}, fmt.Errorf("bad clause atom %q", a)
		}
		side := byte('t')
		if m[1] == "our_bid" {
			side = 'u'
		}
		return patAtom{kind: atomBidcmp, a: side, call: int(id),
			negated: negated}, nil
	}
	if c, ok := condWords[a]; ok {
		return patAtom{kind: atomCond, cond: c, negated: negated}, nil
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
	var toks []string
	for _, t := range strings.Split(pat, "-") {
		if strings.TrimSpace(t) != "" {
			toks = append(toks, strings.TrimSpace(t))
		}
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("empty pattern: %s", raw)
	}
	e := &patEntry{raw: raw, pat: pat, desc: desc, weight: 1.0}
	allAny := true
	for _, t := range toks {
		s, err := parseStep(t)
		if err != nil {
			return nil, err
		}
		if s.kind != stepAny {
			allAny = false
		}
		e.steps = append(e.steps, s)
	}
	if allAny {
		return nil, fmt.Errorf("pattern is only wildcards: %s", raw)
	}
	e.anchored = e.steps[0].kind != stepAny
	groups, err := parseAtoms(clause)
	if err != nil {
		return nil, err
	}
	e.atomGroups = groups

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
		switch strings.Trim(w, ",:;.") {
		case "F":
			e.flagF = true
		case "GF":
			e.flagGF = true
		case "TO":
			e.flagTO = true
		case "PENALTY":
			e.flagPen = true
		}
	}
	if e.flagTO && e.flagPen {
		return nil, fmt.Errorf("entry is both TO and PENALTY: %s", raw)
	}
	return e, nil
}

// ---- Matching ----

func callMatchesStep(step patStep, call int, binds map[byte]int) bool {
	switch step.kind {
	case stepPass:
		return call == int(callPass)
	case stepDbl:
		return call == int(callX)
	case stepRdbl:
		return call == int(callXX)
	}
	if call < firstBid {
		return false // negated steps match bids only
	}
	level := (call-firstBid)/5 + 1
	denom := (call - firstBid) % 5
	if step.level != 0 && level != step.level {
		return false
	}
	d := step.denom
	ok := false
	switch {
	case d == 'N':
		ok = denom == denomN
	case isVarChar(d):
		if denom != denomN { // variables are suits, never NT
			if prev, bound := binds[d]; bound {
				ok = prev == denom
			} else {
				distinct := true
				for _, v := range binds {
					if v == denom {
						distinct = false
						break
					}
				}
				if distinct {
					binds[d] = denom
					ok = true
				}
			}
		}
	default:
		ok = denom == denomOfChar(d)
	}
	if step.negated {
		return !ok
	}
	return ok
}

// matchFrom binds e.steps[stepI:] onto seq[pos:], consuming exactly through
// the candidate (the last seq element); nil when no alignment works. "..."
// steps absorb any run of calls (lazily), so matching backtracks.
func matchFrom(e *patEntry, dealer int, seq []int, us, stepI, pos int,
	binds map[byte]int) map[byte]int {
	if stepI == len(e.steps) {
		if pos == len(seq) {
			return binds
		}
		return nil
	}
	step := e.steps[stepI]
	if step.kind == stepAny {
		for skip := 0; skip <= len(seq)-pos; skip++ {
			cp := make(map[byte]int, len(binds)+4)
			for k, v := range binds {
				cp[k] = v
			}
			if r := matchFrom(e, dealer, seq, us, stepI+1, pos+skip, cp); r != nil {
				return r
			}
		}
		return nil
	}
	i := pos
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
		return nil
	}
	seat := (dealer + i) % 4
	if step.them != (seat%2 != us) {
		return nil
	}
	if !callMatchesStep(step, seq[i], binds) {
		return nil
	}
	return matchFrom(e, dealer, seq, us, stepI+1, i+1, binds)
}

func denomRef(c byte, binds map[byte]int) int {
	if isVarChar(c) || c == 'T' || c == 'P' || c == 'L' || c == 'O' {
		if v, ok := binds[c]; ok {
			return v
		}
		return -1
	}
	return denomOfChar(c)
}

// matchState is the auction state the clause conditions read, built by
// replaying the calls from the dealer (the Go twin of patterns.MatchState,
// limited to what matching and rendering need: the force/double discipline
// fields are training-only).
type matchState struct {
	gf             [2]bool // file GF latches OR the structural game force
	gfFile         [2]bool
	dbls           [2]int
	bids           [2]int
	xx             [2]bool
	high           [2]int
	activity       [2]int
	passesSinceBid int
	called         [4]bool
	bidSeat        [4]bool
	reader         *auctionReader // parallel reader (roles, structural gf)
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
		st.bids[side]++
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

func condOk(cond int, st *matchState, info callInfo, r *auctionReader,
	us, seat int) bool {
	partner := (seat + 2) % 4
	switch cond {
	case condGF:
		return st.gf[us]
	case condUncontested:
		return st.activity[1-us] == 0
	case condContested:
		return st.activity[1-us] > 0
	case condBalancing:
		return st.passesSinceBid >= 2
	case condPassed:
		return st.called[seat] && !st.bidSeat[seat]
	case condFirst:
		return !r.called[seat]
	case condOpener:
		return r.opener == seat
	case condPartnerbid:
		return r.seatBid(partner)
	case condNewsuit:
		return info.kind == kindBid && info.denom < denomN &&
			r.seatSuits[seat]&(1<<uint(info.denom)) == 0
	case condWebid:
		return r.sideHasBid[us]
	case condNotopening:
		return r.highest > 0
	case condNoxyz:
		return r.xyzStage == xyzNone
	case condNostrong2c:
		return r.strong2cSide != us
	case condFourth:
		return info.fourthSuit
	case condWeXX:
		return st.xx[us]
	case condTheyXX:
		return st.xx[1-us]
	}
	return false
}

func numCmp(v, n int, op string) bool {
	switch op {
	case ">":
		return v > n
	case ">=":
		return v >= n
	case "<":
		return v < n
	case "<=":
		return v <= n
	default:
		return v == n
	}
}

func atomsOk(groups [][]patAtom, binds map[byte]int, st *matchState,
	info callInfo, r *auctionReader, us, seat int) bool {
	if len(groups) == 0 {
		return true
	}
	for _, g := range groups { // OR over groups
		ok := true
		for _, a := range g { // AND within a group
			var res bool
			switch a.kind {
			case atomCmp:
				x, y := denomRef(a.a, binds), denomRef(a.b, binds)
				switch a.op {
				case "<", ">", ">=", "<=":
					res = x >= 0 && y >= 0 && numCmp(x, y, a.op)
				case "=":
					res = x >= 0 && y >= 0 && x == y
				default: // !=
					res = x < 0 || y < 0 || x != y
				}
			case atomCnt:
				var n int
				if a.cntWhat == 'd' {
					n = st.dbls[us]
					if a.a == 't' {
						n = st.dbls[1-us]
					}
				} else {
					n = st.bids[us]
					if a.a == 't' {
						n = st.bids[1-us]
					}
				}
				res = numCmp(n, a.n, a.op)
			case atomNum:
				if info.kind != kindBid {
					res = false
				} else {
					v := info.jump
					if a.cntWhat == 'l' {
						v = info.level
					}
					res = numCmp(v, a.n, a.op)
				}
			case atomBidcmp:
				h := st.high[1-us]
				if a.a == 'u' {
					h = st.high[us]
				}
				res = h >= a.call
			case atomBelow:
				res = info.kind == kindBid && info.callID < a.call
			default: // atomCond
				res = condOk(a.cond, st, info, r, us, seat)
			}
			if a.negated {
				res = !res
			}
			if !res {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// contextBinds adds the context suit refs T/P/L/O (our last suit, partner's
// last suit, their last suit, the other major of the candidate's major).
func contextBinds(binds map[byte]int, r *auctionReader, info callInfo,
	seat int) map[byte]int {
	side := seat % 2
	partner := (seat + 2) % 4
	out := make(map[byte]int, len(binds)+4)
	for k, v := range binds {
		out[k] = v
	}
	if t := r.lastSuitOf(side, -1); t >= 0 {
		out['T'] = t
	}
	if p := r.lastSuitOf(side, partner); p >= 0 {
		out['P'] = p
	}
	if l := r.lastSuitOf(1-side, -1); l >= 0 {
		out['L'] = l
	}
	if info.kind == kindBid && isMajor(info.denom) {
		other := 2 // H
		if info.denom == 2 {
			other = 3 // S
		}
		out['O'] = other
	}
	return out
}

// matchFirst returns the first entry (file order) matching `action` after
// `calls`, with its suit-variable and context bindings; (nil, nil) when no
// entry matches.
func matchFirst(entries []*patEntry, dealer int, calls []int, action int,
	st *matchState) (*patEntry, map[byte]int) {
	seq := make([]int, 0, len(calls)+1)
	seq = append(seq, calls...)
	seq = append(seq, action)
	seat := (dealer + len(seq) - 1) % 4
	us := seat % 2
	info := st.reader.classify(action, seat)
	for _, e := range entries {
		var starts []int
		if !e.anchored {
			starts = []int{0} // the leading "..." absorbs anything
		} else if e.steps[0].kind == stepPass {
			starts = []int{0}
		} else {
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
			binds := matchFrom(e, dealer, seq, us, 0, s, map[byte]int{})
			if binds == nil {
				continue
			}
			full := contextBinds(binds, st.reader, info, seat)
			if atomsOk(e.atomGroups, full, st, info, st.reader, us, seat) {
				return e, full
			}
		}
	}
	return nil, nil
}

// ---- Alert rendering (mirrors patterns.render_desc) ----

var kcText = map[string]string{
	"0/3": "0 or 3 keycards",
	"1/4": "1 or 4 keycards",
	"2-":  "2 keycards without the trump queen",
	"2+":  "2 keycards with the trump queen",
}

var keywordText = map[string]string{
	"F": "forcing", "GF": "game force", "TO": "takeout",
	"PENALTY": "penalty", "OR": "or", "openp": "opening values",
	"no4M": "no biddable 4-card major", "noLong5M": "no longer 5-card major",
	"stop": "stopper in their suit", "2top": "2 of the top 3 honours",
}

func suitSym(ref byte, binds map[byte]int) string {
	d := denomRef(ref, binds)
	if d >= 0 && d < denomN {
		return suitSymbols[d]
	}
	return "any"
}

func prettyToken(w string, binds map[byte]int) string {
	word := strings.Trim(w, ",:;.")
	if m := reLen.FindStringSubmatch(word); m != nil {
		var sym string
		if m[4] == "any" {
			sym = "any"
		} else {
			sym = suitSym(m[4][0], binds)
		}
		if m[2] == "+" {
			return m[1] + "+ " + sym
		}
		if strings.HasPrefix(m[2], "-") {
			return m[1] + "-" + m[3] + " " + sym
		}
		return m[1] + " " + sym
	}
	if m := rePts.FindStringSubmatch(word); m != nil {
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
	if m := reKC.FindStringSubmatch(word); m != nil {
		return kcText[m[1]]
	}
	if reBrace.MatchString(word) {
		return reBrace.ReplaceAllStringFunc(word, func(s string) string {
			return suitSym(s[1], binds)
		})
	}
	if t, ok := keywordText[word]; ok {
		return t
	}
	return word
}

// renderDesc is the alert text of an entry: suit variables and {T}/{P}/{L}/{O}
// refs resolved to symbols, enforcement tokens prettified, @attributes
// dropped.
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
