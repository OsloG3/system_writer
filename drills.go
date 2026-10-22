package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---- Card problems: authored card-play drills ----
//
// A problem is a fully scripted board -- deal, auction, comments and every
// card of the play, each hero card carrying the author's "why" message --
// stored per problem in data/drills/<id>.json. Nothing here touches the
// bidding-dt sidecar: legality and scoring are all local.
//
// Practice replays the script: the hero (the declarer side for declarer
// problems, one chosen defender for defense problems) must find the authored
// card at every turn; a different pick is refused with a plain 409 (the
// client buzzes and flashes red) and nothing is revealed. The message the
// author attached to the correct card is an opt-in HINT: the practitioner
// asks for it explicitly (GET /api/drill/session/{sid}/hint), and every
// message is shown again in the end-of-hand review, together with the
// problem's optional explanation. Every other seat plays itself from the
// script. The hand ends after 52 cards or at the author's claim (a trick
// count plus a plain final score -- no IMPs, no par).

const (
	drillMaxCalls   = 40 // authored auctions are not bound by the model's cap
	drillMaxLines   = 8  // alternative lines per problem
	drillSessionTTL = 6 * time.Hour
)

var (
	drillsMu sync.Mutex
	drills   = map[string]*DrillProblem{} // keyed by problem ID

	drillSessMu sync.Mutex
	drillSess   = map[string]*DrillSession{}
)

// DrillPlay is one scripted card. Msg and Alts are only allowed on the seats
// the hero controls. Alts lists equivalent cards: interchangeable with Card
// (and with each other) for the rest of the line, as long as swapping them
// never changes who wins a trick -- validated on save. When the swap WOULD
// change a trick winner the author must give separate lines instead.
type DrillPlay struct {
	Seat int      `json:"seat"`
	Card string   `json:"card"`
	Msg  string   `json:"msg,omitempty"`
	Alts []string `json:"alts,omitempty"`
}

// DrillClaim ends a line early: the declarer's final trick count. The result
// is shown as the contract and its margin (e.g. "4S-1") -- no score, no IMPs.
type DrillClaim struct {
	Tricks int `json:"tricks"`
}

// DrillBotEnd hands the rest of the hand to the card-play bots: from the end
// of the script the hero may play ANY legal card and every other seat is
// played by the bidding-dt sidecar. The counted result is graded against the
// problem's Target like every other ending.
type DrillBotEnd struct {
	// Expect is the legacy per-line target (declarer tricks), folded into
	// DrillProblem.Target by migrateTarget and never written back.
	Expect int `json:"expect,omitempty"`
}

// DrillLine is one full scripted play with its ending. A problem carries one
// line, or several when a hero choice changes who wins a trick -- each such
// choice starts its own line and practice follows whichever one matches the
// cards the hero actually plays.
type DrillLine struct {
	Plays []DrillPlay  `json:"plays"`
	Claim *DrillClaim  `json:"claim,omitempty"` // nil Claim + nil Bot = all 52 played out
	Bot   *DrillBotEnd `json:"bot,omitempty"`

	classes map[int]map[string]string // seat -> card -> union-find parent (built by validate)
}

type DrillProblem struct {
	ID             string      `json:"id"`
	Owner          string      `json:"owner"`
	Title          string      `json:"title,omitempty"`
	Role           string      `json:"role"`     // "declarer" | "defense"
	HeroSeat       int         `json:"heroSeat"` // declarer, or the one chosen defender
	Difficulty     int         `json:"difficulty"`
	Dealer         int         `json:"dealer"`
	Vuln           int         `json:"vuln"`
	Hands          [4]string   `json:"hands"` // N,E,S,W 'S.H.D.C'
	Calls          []string    `json:"calls"`
	BidComment     string      `json:"bidComment,omitempty"`
	ProblemComment string      `json:"problemComment,omitempty"`
	// Explanation is the afterword of the problem: unlike ProblemComment
	// (part of the pre-play briefing) it travels to the client only once the
	// hand is over, as the end-of-hand explication of the whole deal.
	Explanation string `json:"explanation,omitempty"`
	// Target is the number of tricks the hero's side must take (or beat) for
	// a practice run to count as a success: declarer tricks on a declarer
	// problem, defenders' tricks on a defense one. It is shown before the
	// play starts and grades every ending (claim, bots, played out).
	Target int `json:"target"`
	// AllowWrong lets the practitioner stray from every scripted line: the
	// wrong card stands, the bots finish the hand and it is graded against
	// Target, instead of the pick being refused with a hint. This is how an
	// author opens up incorrect lines without scripting them.
	AllowWrong bool        `json:"allowWrong,omitempty"`
	Lines      []DrillLine `json:"lines"`
	// Ratings maps username -> stars (1..5), given after finishing a
	// practice run; the library shows the average and sorts by it.
	Ratings map[string]int `json:"ratings,omitempty"`
	// Played maps username -> timestamp of their first finished practice
	// run. Server-side only (like Ratings); drives the played/unplayed
	// filter in the library.
	Played map[string]string `json:"played,omitempty"`
	// Legacy single-line shape (before Lines existed); migrated on load and
	// on every incoming payload, never written back.
	Plays   []DrillPlay `json:"plays,omitempty"`
	Claim   *DrillClaim `json:"claim,omitempty"`
	Created string      `json:"created"`
	Updated string      `json:"updated"`
}

// ratingSummary averages a problem's stars. Callers must hold drillsMu.
func ratingSummaryLocked(p *DrillProblem) (avg float64, count int) {
	count = len(p.Ratings)
	if count == 0 {
		return 0, 0
	}
	sum := 0
	for _, v := range p.Ratings {
		sum += v
	}
	return math.Round(float64(sum)/float64(count)*100) / 100, count
}

// migrateLines folds the legacy single Plays+Claim shape into Lines.
func (p *DrillProblem) migrateLines() {
	if len(p.Lines) == 0 && len(p.Plays) > 0 {
		p.Lines = []DrillLine{{Plays: p.Plays, Claim: p.Claim}}
	}
	p.Plays, p.Claim = nil, nil
}

// heroSide converts a declarer trick count to the hero's side of the table.
func (p *DrillProblem) heroSide(decl int) int {
	if p.Role == "declarer" {
		return decl
	}
	return 13 - decl
}

// clearLegacyExpect drops the per-line bot targets problems carried before
// the problem-level Target existed.
func (p *DrillProblem) clearLegacyExpect() {
	for i := range p.Lines {
		if p.Lines[i].Bot != nil {
			p.Lines[i].Bot.Expect = 0
		}
	}
}

// migrateTarget fills Target for problems saved before it existed, from the
// first line's ending: a claim's tricks or the old bot expectation. A fully
// scripted line (or a bot end without an expectation) is counted later, in
// validateDrill, once the play is simulated.
func (p *DrillProblem) migrateTarget() {
	defer p.clearLegacyExpect()
	if p.Target > 0 || len(p.Lines) == 0 {
		return
	}
	l := p.Lines[0]
	switch {
	case l.Claim != nil:
		p.Target = p.heroSide(l.Claim.Tricks)
	case l.Bot != nil && l.Bot.Expect > 0:
		p.Target = p.heroSide(l.Bot.Expect)
	}
}

// DrillResult grades a finished run against the problem's Target: Success
// when the hero's side took at least Target tricks, however the hand ended.
type DrillResult struct {
	Contract *PlayContract `json:"contract"`
	Tricks   int           `json:"tricks"` // declarer tricks taken
	Claimed  bool          `json:"claimed"`
	Wrong    int           `json:"wrong"`
	Botted   bool          `json:"botted,omitempty"` // the bots finished the hand
	Target   int           `json:"target"`           // hero-side tricks needed
	Success  bool          `json:"success"`          // hero's side took at least Target
}

// DrillSession is one practice run. The problem is snapshotted at the start
// so edits cannot desync a session in flight. In-memory only, like the solo
// play boards.
type DrillSession struct {
	mu         sync.Mutex
	ID         string
	User       string
	Problem    DrillProblem
	contract   *PlayContract
	trump      byte
	Plays      []DrillPlay // applied prefix of the script (hero cards as actually played)
	Live       []int       // indices of the lines still matching the play so far
	Bot        bool        // the scripted part is over; bots play every non-hero seat
	DeclTricks int
	Wrong      int
	Done       bool
	Result     *DrillResult
	Created    time.Time
}

func drillsDir() string          { return filepath.Join(dataDir, "drills") }
func drillPath(id string) string { return filepath.Join(drillsDir(), id+".json") }

func initDrills() {
	if err := os.MkdirAll(drillsDir(), 0o755); err != nil {
		log.Printf("Could not create drills directory: %v", err)
		return
	}
	files, err := filepath.Glob(filepath.Join(drillsDir(), "*.json"))
	if err != nil {
		return
	}
	n := 0
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var p DrillProblem
		if err := json.Unmarshal(data, &p); err != nil {
			log.Printf("Could not parse %s: %v", f, err)
			continue
		}
		if p.ID == "" {
			p.ID = strings.TrimSuffix(filepath.Base(f), ".json")
		}
		if !validID(p.ID) {
			continue
		}
		p.migrateLines()
		p.migrateTarget()
		drills[p.ID] = &p
		n++
	}
	if n > 0 {
		log.Printf("Loaded %d card problem(s)", n)
	}
}

// saveDrillLocked persists a problem. Callers must hold drillsMu.
func saveDrillLocked(p *DrillProblem) error {
	return writeJSONFile(drillPath(p.ID), p)
}

func lookupDrill(id string) *DrillProblem {
	drillsMu.Lock()
	defer drillsMu.Unlock()
	return drills[id]
}

// ---- Validation ----

func validSeat(s int) bool { return s >= 0 && s <= 3 }

// validateHands checks the four 'S.H.D.C' strings form a full 52-card deck
// with 13 cards per seat.
func validateHands(hands [4]string) error {
	seen := map[string]bool{}
	for seat, h := range hands {
		parts := strings.Split(h, ".")
		if len(parts) != 4 {
			return fmt.Errorf("hand %s: want four dot-separated suits (S.H.D.C)", "NESW"[seat:seat+1])
		}
		n := 0
		for s, suit := range parts {
			for _, r := range suit {
				card := string(r) + string("SHDC"[s])
				if normalizeCard(card) != card {
					return fmt.Errorf("hand %s: bad card %q", "NESW"[seat:seat+1], card)
				}
				if seen[card] {
					return fmt.Errorf("card %s appears twice", card)
				}
				seen[card] = true
				n++
			}
		}
		if n != 13 {
			return fmt.Errorf("hand %s has %d cards, want 13", "NESW"[seat:seat+1], n)
		}
	}
	if len(seen) != 52 {
		return errors.New("the four hands must cover all 52 cards")
	}
	return nil
}

func bidRank(c string) int {
	if len(c) == 2 && c[0] >= '1' && c[0] <= '7' && strings.IndexByte("CDHSN", c[1]) >= 0 {
		return int(c[0]-'0')*10 + strings.IndexByte("CDHSN", c[1])
	}
	return -1
}

// validateAuction checks an authored auction is legal and ends in a contract
// (three passes over a bid; passed-out problems are not card-play problems).
// Returns the resulting contract.
func validateAuction(dealer int, calls []string) (*PlayContract, error) {
	if len(calls) < 4 {
		return nil, errors.New("the auction is too short")
	}
	if len(calls) > drillMaxCalls {
		return nil, fmt.Errorf("the auction has more than %d calls", drillMaxCalls)
	}
	lastBidRank, lastBidSeat := -1, -1
	passRun := 0 // consecutive passes immediately before the current call
	for i, raw := range calls {
		c := normalizeCall(raw)
		if c == "" {
			return nil, fmt.Errorf("call %d is not a valid call", i+1)
		}
		if passRun >= 3 {
			return nil, errors.New("the auction continues after three passes")
		}
		seat := (dealer + i) % 4
		// the last call that was not a pass: a double or redouble may have
		// passes in between (Law 19 -- balancing doubles, a late redouble)
		// but no other call
		prevNP := ""
		for j := i - 1; j >= 0; j-- {
			if pc := normalizeCall(calls[j]); pc != "P" {
				prevNP = pc
				break
			}
		}
		switch {
		case c == "P":
		case c == "X":
			// a double over an opponent's bid, only passes in between
			if bidRank(prevNP) < 0 || lastBidSeat < 0 || lastBidSeat%2 == seat%2 {
				return nil, fmt.Errorf("call %d: double must be over an opponent's bid, with only passes in between", i+1)
			}
		case c == "XX":
			// a redouble over the opponents' double of your side's bid,
			// only passes in between
			if prevNP != "X" || lastBidSeat < 0 || lastBidSeat%2 != seat%2 {
				return nil, fmt.Errorf("call %d: redouble must be over the opponents' double of your bid, with only passes in between", i+1)
			}
		default: // a bid
			if bidRank(c) <= lastBidRank {
				return nil, fmt.Errorf("call %d: %s does not outrank the previous bid", i+1, c)
			}
			lastBidRank, lastBidSeat = bidRank(c), seat
		}
		if c == "P" {
			passRun++
		} else {
			passRun = 0
		}
	}
	if !auctionFinished(calls) {
		return nil, errors.New("the auction must end with a contract and three passes")
	}
	contract := contractFromCalls(dealer, calls)
	if contract == nil {
		return nil, errors.New("the auction must end in a contract, not a pass-out")
	}
	return contract, nil
}

// heroSeatsLocked lists the seats the hero plays for a problem.
func (p *DrillProblem) heroSeats() []int {
	if p.Role == "declarer" {
		return []int{p.HeroSeat, (p.HeroSeat + 2) % 4}
	}
	return []int{p.HeroSeat}
}

func seatName(seat int) string { return string("NESW"[seat : seat+1]) }

// simulateLine walks one scripted line, enforcing turn order, card ownership
// and follow-suit. It returns the winner of every completed trick (in trick
// order) and the declarer side's trick count.
func simulateLine(p *DrillProblem, contract *PlayContract, plays []DrillPlay) ([]int, int, error) {
	dummy := (contract.Declarer + 2) % 4
	remaining := map[int]map[string]bool{}
	for seat := 0; seat < 4; seat++ {
		remaining[seat] = map[string]bool{}
		for _, c := range handCards(p.Hands[seat]) {
			remaining[seat][c] = true
		}
	}
	var winners []int
	declTricks := 0
	next := (contract.Declarer + 1) % 4
	for i, dp := range plays {
		card := normalizeCard(dp.Card)
		if card == "" {
			return nil, 0, fmt.Errorf("play %d: bad card %q", i+1, dp.Card)
		}
		if !validSeat(dp.Seat) {
			return nil, 0, fmt.Errorf("play %d: bad seat", i+1)
		}
		if dp.Seat != next {
			return nil, 0, fmt.Errorf("play %d: %s plays out of turn (expected %s)",
				i+1, seatName(dp.Seat), seatName(next))
		}
		if !remaining[dp.Seat][card] {
			return nil, 0, fmt.Errorf("play %d: %s does not hold %s", i+1, seatName(dp.Seat), card)
		}
		if i%4 > 0 {
			lead := cardSuit(plays[(i/4)*4].Card)
			if cardSuit(card) != lead {
				hasLead := false
				for c := range remaining[dp.Seat] {
					if cardSuit(c) == lead {
						hasLead = true
						break
					}
				}
				if hasLead {
					return nil, 0, fmt.Errorf("play %d: %s must follow suit", i+1, seatName(dp.Seat))
				}
			}
		}
		delete(remaining[dp.Seat], card)
		if (i+1)%4 == 0 {
			trick := make([]PlayCard, 4)
			for j := 0; j < 4; j++ {
				trick[j] = PlayCard{Seat: plays[i-3+j].Seat, Card: plays[i-3+j].Card}
			}
			w := trickWinner(trick, contract.Denom[0])
			winners = append(winners, w)
			if w == contract.Declarer || w == dummy {
				declTricks++
			}
			next = w
		} else {
			next = (dp.Seat + 1) % 4
		}
	}
	return winners, declTricks, nil
}

// buildClasses unions each hero play's card with its declared alternatives,
// per seat. Two plays sharing a card merge their classes, so declaring the
// alternatives once anywhere in the line covers the whole line: after the
// hero swaps within a class, every later class member they still hold stays
// acceptable.
func (l *DrillLine) buildClasses(hero []int) {
	parent := map[int]map[string]string{}
	find := func(seat int, c string) string {
		m := parent[seat]
		if m == nil {
			m = map[string]string{}
			parent[seat] = m
		}
		if _, ok := m[c]; !ok {
			m[c] = c
		}
		for m[c] != c {
			c = m[c]
		}
		return c
	}
	for _, dp := range l.Plays {
		if !slices.Contains(hero, dp.Seat) || len(dp.Alts) == 0 {
			continue
		}
		root := find(dp.Seat, dp.Card)
		for _, a := range dp.Alts {
			ra := find(dp.Seat, a)
			if ra != root {
				parent[dp.Seat][ra] = root
			}
		}
	}
	l.classes = parent
}

// classRoot returns the equivalence-class representative of a card for a
// seat; a card in no declared class is its own root.
func (l *DrillLine) classRoot(seat int, card string) string {
	m := l.classes[seat]
	c := card
	if _, ok := m[c]; !ok {
		return card
	}
	for m[c] != c {
		c = m[c]
	}
	return c
}

// classMembers lists every card equivalent to `card` for `seat` (including
// the card itself), sorted; a card in no class returns just itself.
func (l *DrillLine) classMembers(seat int, card string) []string {
	if l.classes == nil || l.classes[seat] == nil {
		return []string{card}
	}
	root := l.classRoot(seat, card)
	var out []string
	for c := range l.classes[seat] {
		if l.classRoot(seat, c) == root {
			out = append(out, c)
		}
	}
	if out == nil {
		out = []string{card}
	}
	slices.Sort(out)
	return out
}

// validateEquivalence checks every declared equivalence class really is
// interchangeable: ANY assignment of the class members to the line's scripted
// plays of that class (the hero may swap once and keep swapping later) must
// stay legal and leave EVERY trick winner unchanged. A class whose members
// can move a trick winner means the choice matters -- the author has to give
// separate lines for it instead.
func validateEquivalence(p *DrillProblem, contract *PlayContract, l *DrillLine, baseWinners []int) error {
	hero := p.heroSeats()
	type classKey struct {
		seat int
		root string
	}
	checked := map[classKey]bool{}
	for _, dp := range l.Plays {
		if !slices.Contains(hero, dp.Seat) {
			continue
		}
		k := classKey{dp.Seat, l.classRoot(dp.Seat, dp.Card)}
		if checked[k] {
			continue
		}
		members := l.classMembers(dp.Seat, dp.Card)
		if len(members) < 2 {
			continue // no equivalence declared for this card
		}
		checked[k] = true
		var positions []int
		for j, q := range l.Plays {
			if q.Seat == dp.Seat && l.classRoot(dp.Seat, q.Card) == k.root {
				positions = append(positions, j)
			}
		}
		if err := validateClassAssignments(p, contract, l, baseWinners, positions, members); err != nil {
			return fmt.Errorf("%s's equivalent cards (%s) are not interchangeable: %v -- give separate lines for that choice",
				seatName(dp.Seat), strings.Join(cardNames(members), ", "), err)
		}
	}
	return nil
}

func cardNames(cards []string) []string {
	out := make([]string, len(cards))
	for i, c := range cards {
		out[i] = drillCardName(c)
	}
	return out
}

// validateClassAssignments tries every injective assignment of class members
// onto the class's scripted positions and re-simulates the line, requiring
// identical trick winners. Bounded: an impractically large class is rejected
// rather than exploded.
func validateClassAssignments(p *DrillProblem, contract *PlayContract, l *DrillLine, baseWinners []int, positions []int, members []string) error {
	perms := 1
	for k := 0; k < len(positions); k++ {
		if len(members)-k <= 0 {
			break
		}
		perms *= len(members) - k
		if perms > 5000 {
			return errors.New("the group is too large to verify -- narrow the alternatives")
		}
	}
	used := make([]bool, len(members))
	pick := make([]string, len(positions))
	var rec func(k int) error
	rec = func(k int) error {
		if k == len(positions) {
			swapped := make([]DrillPlay, len(l.Plays))
			copy(swapped, l.Plays)
			for idx, pos := range positions {
				swapped[pos].Card = pick[idx]
			}
			winners, _, err := simulateLine(p, contract, swapped)
			if err != nil {
				return err
			}
			for t, w := range winners {
				if t < len(baseWinners) && w != baseWinners[t] {
					return fmt.Errorf("the winner of trick %d would change", t+1)
				}
			}
			return nil
		}
		for mi, m := range members {
			if used[mi] {
				continue
			}
			used[mi] = true
			pick[k] = m
			if err := rec(k + 1); err != nil {
				used[mi] = false
				return err
			}
			used[mi] = false
		}
		return nil
	}
	return rec(0)
}

// validateLine checks one line: its plays, the declared equivalences and its
// ending (played out / claim / bots). Returns the declarer tricks the script
// itself won.
func validateLine(p *DrillProblem, contract *PlayContract, l *DrillLine, idx int) (int, error) {
	hero := p.heroSeats()
	if len(l.Plays) == 0 && l.Bot == nil {
		return 0, fmt.Errorf("line %d: record the card play first", idx+1)
	}
	if len(l.Plays) > 52 {
		return 0, fmt.Errorf("line %d: a line cannot have more than 52 plays", idx+1)
	}
	held := [4][]string{}
	for seat := 0; seat < 4; seat++ {
		held[seat] = handCards(p.Hands[seat])
	}
	for i := range l.Plays {
		dp := &l.Plays[i]
		if c := normalizeCard(dp.Card); c != "" {
			dp.Card = c
		}
		isHero := slices.Contains(hero, dp.Seat)
		if strings.TrimSpace(dp.Msg) != "" && !isHero {
			return 0, fmt.Errorf("line %d play %d: only the hero's cards may carry a message", idx+1, i+1)
		}
		for j, a := range dp.Alts {
			an := normalizeCard(a)
			if an == "" {
				return 0, fmt.Errorf("line %d play %d: bad alternative card %q", idx+1, i+1, a)
			}
			dp.Alts[j] = an
			switch {
			case !isHero:
				return 0, fmt.Errorf("line %d play %d: only the hero's cards may have alternatives", idx+1, i+1)
			case an == dp.Card:
				return 0, fmt.Errorf("line %d play %d: %s is the played card itself", idx+1, i+1, an)
			case slices.Contains(dp.Alts[:j], an):
				return 0, fmt.Errorf("line %d play %d: duplicate alternative %s", idx+1, i+1, an)
			case !validSeat(dp.Seat) || !slices.Contains(held[dp.Seat], an):
				return 0, fmt.Errorf("line %d play %d: %s does not hold %s", idx+1, i+1, seatName(dp.Seat), an)
			}
		}
		if len(dp.Alts) == 0 {
			dp.Alts = nil
		}
	}
	winners, declTricks, err := simulateLine(p, contract, l.Plays)
	if err != nil {
		return 0, fmt.Errorf("line %d: %v", idx+1, err)
	}
	l.buildClasses(hero)
	if err := validateEquivalence(p, contract, l, winners); err != nil {
		return 0, fmt.Errorf("line %d: %v", idx+1, err)
	}
	switch {
	case l.Claim != nil && l.Bot != nil:
		return 0, fmt.Errorf("line %d: a line ends with a claim OR with the bots, not both", idx+1)
	case l.Claim != nil:
		if len(l.Plays)%4 != 0 || len(l.Plays) < 4 {
			return 0, fmt.Errorf("line %d: a claim must come at the end of a complete trick", idx+1)
		}
		remaining := (52 - len(l.Plays)) / 4
		if l.Claim.Tricks < declTricks || l.Claim.Tricks > declTricks+remaining {
			return 0, fmt.Errorf("line %d: the claimed %d tricks are impossible (%d already won, %d still to play)",
				idx+1, l.Claim.Tricks, declTricks, remaining)
		}
	case l.Bot != nil:
		if len(l.Plays) >= 52 {
			return 0, fmt.Errorf("line %d: there is nothing left for the bots to play", idx+1)
		}
	default:
		if len(l.Plays) != 52 {
			return 0, fmt.Errorf("line %d: the play must cover all 52 cards, or end with a claim or with the bots", idx+1)
		}
	}
	return declTricks, nil
}

// validateLineDivergence keeps two lines unambiguous: while both are still
// in play they must agree on every auto-seat card, and they must part ways
// on a genuinely different hero card (not two interchangeable ones -- those
// belong in one line as alternatives).
func validateLineDivergence(p *DrillProblem, l1, l2 *DrillLine, i, j int) error {
	hero := p.heroSeats()
	n := min(len(l1.Plays), len(l2.Plays))
	for k := 0; k < n; k++ {
		a, b := l1.Plays[k], l2.Plays[k]
		if a.Seat != b.Seat {
			return fmt.Errorf("lines %d and %d: the turn order differs at play %d before any hero choice -- the seats must match until the hero's cards do", i+1, j+1, k+1)
		}
		if a.Card == b.Card && slices.Equal(l1.classMembers(a.Seat, a.Card), l2.classMembers(b.Seat, b.Card)) {
			continue
		}
		if !slices.Contains(hero, a.Seat) {
			return fmt.Errorf("lines %d and %d: they differ at play %d on %s's card, but only the hero's choices may branch a line", i+1, j+1, k+1, seatName(a.Seat))
		}
		if l1.classRoot(a.Seat, a.Card) == l1.classRoot(a.Seat, b.Card) ||
			l2.classRoot(a.Seat, a.Card) == l2.classRoot(a.Seat, b.Card) {
			return fmt.Errorf("lines %d and %d: %s and %s at play %d are interchangeable -- keep one line and mark them as alternatives instead",
				i+1, j+1, drillCardName(a.Card), drillCardName(b.Card), k+1)
		}
		return nil // a real divergence: from here the lines may differ freely
	}
	return fmt.Errorf("lines %d and %d never diverge on a hero card -- merge them, or extend the shorter one", i+1, j+1)
}

// validateDrill checks a problem as authored: the deal, the auction, the hero
// setup and every line of the scripted play with its ending.
func validateDrill(p *DrillProblem) (*PlayContract, error) {
	if p.Role != "declarer" && p.Role != "defense" {
		return nil, errors.New(`role must be "declarer" or "defense"`)
	}
	if p.Difficulty < 1 || p.Difficulty > 5 {
		return nil, errors.New("difficulty must be between 1 and 5 stars")
	}
	if !validSeat(p.Dealer) || !validSeat(p.Vuln) || !validSeat(p.HeroSeat) {
		return nil, errors.New("dealer, vulnerability and hero seat must be valid seats")
	}
	if err := validateHands(p.Hands); err != nil {
		return nil, err
	}
	for i, c := range p.Calls {
		n := normalizeCall(c)
		if n == "" {
			return nil, fmt.Errorf("call %d is not a valid call", i+1)
		}
		p.Calls[i] = n
	}
	contract, err := validateAuction(p.Dealer, p.Calls)
	if err != nil {
		return nil, err
	}
	defenders := []int{(contract.Declarer + 1) % 4, (contract.Declarer + 3) % 4}
	if p.Role == "declarer" {
		if p.HeroSeat != contract.Declarer {
			return nil, errors.New("a declarer problem's hero must be the declarer")
		}
	} else if !slices.Contains(defenders, p.HeroSeat) {
		return nil, errors.New("a defense problem's hero must be one of the defenders")
	}
	p.migrateLines()
	p.migrateTarget()
	if len(p.Lines) == 0 {
		return nil, errors.New("record the card play first")
	}
	if len(p.Lines) > drillMaxLines {
		return nil, fmt.Errorf("a problem may have at most %d lines", drillMaxLines)
	}
	scripted := 0
	for i := range p.Lines {
		decl, err := validateLine(p, contract, &p.Lines[i], i)
		if err != nil {
			return nil, err
		}
		if i == 0 {
			scripted = decl
		}
	}
	if p.Target <= 0 {
		// legacy problem whose first line plays out (or bots without an old
		// expectation): count what the script itself takes
		p.Target = p.heroSide(scripted)
	}
	if p.Target < 1 || p.Target > 13 {
		return nil, errors.New("the target must be between 1 and 13 tricks")
	}
	for i := 0; i < len(p.Lines); i++ {
		for j := i + 1; j < len(p.Lines); j++ {
			if err := validateLineDivergence(p, &p.Lines[i], &p.Lines[j], i, j); err != nil {
				return nil, err
			}
		}
	}
	p.Title = strings.TrimSpace(p.Title)
	p.BidComment = strings.TrimSpace(p.BidComment)
	p.ProblemComment = strings.TrimSpace(p.ProblemComment)
	p.Explanation = strings.TrimSpace(p.Explanation)
	return contract, nil
}

var cardRankNames = map[byte]string{
	'A': "A", 'K': "K", 'Q': "Q", 'J': "J", 'T': "10",
	'9': "9", '8': "8", '7': "7", '6': "6", '5': "5",
	'4': "4", '3': "3", '2': "2",
}

// drillCardName renders a card token for prose, e.g. "AS" -> "A\u2660".
func drillCardName(card string) string {
	if len(card) != 2 {
		return card
	}
	suit := map[byte]string{'S': "\u2660", 'H': "\u2665", 'D': "\u2666", 'C': "\u2663"}[card[1]]
	return cardRankNames[card[0]] + suit
}

// defaultHeroMsg is the message the client pre-fills (and the server falls
// back to) for a hero card the author did not annotate.
func defaultHeroMsg(card string) string {
	return fmt.Sprintf("I just decided to play the %s instead.", drillCardName(card))
}

// ---- Session logic (callers hold s.mu) ----

func (s *DrillSession) heroControls(seat int) bool {
	return slices.Contains(s.Problem.heroSeats(), seat)
}

func (s *DrillSession) nextSeatLocked() int {
	n := len(s.Plays)
	if n == 0 {
		return (s.contract.Declarer + 1) % 4
	}
	if n%4 == 0 {
		trick := make([]PlayCard, 4)
		for j := 0; j < 4; j++ {
			trick[j] = PlayCard{Seat: s.Plays[n-4+j].Seat, Card: s.Plays[n-4+j].Card}
		}
		return trickWinner(trick, s.trump)
	}
	return (s.Plays[n-1].Seat + 1) % 4
}

func (s *DrillSession) remainingHandLocked(seat int) []string {
	played := map[string]bool{}
	for _, dp := range s.Plays {
		if dp.Seat == seat {
			played[dp.Card] = true
		}
	}
	var out []string
	for _, c := range handCards(s.Problem.Hands[seat]) {
		if !played[c] {
			out = append(out, c)
		}
	}
	return out
}

func (s *DrillSession) legalCardsLocked(seat int) []string {
	rem := s.remainingHandLocked(seat)
	n := len(s.Plays)
	if n%4 == 0 {
		return rem
	}
	lead := cardSuit(s.Plays[(n/4)*4].Card)
	var follow []string
	for _, c := range rem {
		if cardSuit(c) == lead {
			follow = append(follow, c)
		}
	}
	if len(follow) > 0 {
		return follow
	}
	return rem
}

func (s *DrillSession) applyLocked(dp DrillPlay) {
	s.Plays = append(s.Plays, dp)
	if len(s.Plays)%4 == 0 {
		n := len(s.Plays)
		trick := make([]PlayCard, 4)
		for j := 0; j < 4; j++ {
			trick[j] = PlayCard{Seat: s.Plays[n-4+j].Seat, Card: s.Plays[n-4+j].Card}
		}
		w := trickWinner(trick, s.trump)
		if w == s.contract.Declarer || w == (s.contract.Declarer+2)%4 {
			s.DeclTricks++
		}
	}
}

// liveLine returns the line the session currently follows (the first one
// still alive). While several are alive they all prescribe the same card for
// every auto seat -- lines may only diverge on a hero choice.
func (s *DrillSession) liveLine() *DrillLine {
	return &s.Problem.Lines[s.Live[0]]
}

// lineMatchesAt reports whether `card` played at position i of the session's
// play is what `line` prescribes there: the scripted card itself, or an
// equivalent one (same union-find class).
func (s *DrillSession) lineMatchesAt(l *DrillLine, i int, seat int, card string) bool {
	if i >= len(l.Plays) || l.Plays[i].Seat != seat {
		return false
	}
	dp := l.Plays[i]
	if dp.Card == card {
		return true
	}
	return l.classRoot(seat, card) == l.classRoot(seat, dp.Card)
}

// lineAccepts reports whether `line` takes `card` from the hero at the
// current position.
func (s *DrillSession) lineAccepts(line int, seat int, card string) bool {
	return s.lineMatchesAt(&s.Problem.Lines[line], len(s.Plays), seat, card)
}

// gradeLocked ends the hand and grades it against the problem's Target: the
// hero's side taking at least Target tricks is a success, however the hand
// ended (claimed, played out, or finished by the bots).
func (s *DrillSession) gradeLocked(tricks int, claimed, botted bool) {
	s.Done = true
	s.Result = &DrillResult{
		Contract: s.contract, Tricks: tricks,
		Claimed: claimed, Wrong: s.Wrong,
		Botted: botted, Target: s.Problem.Target,
		Success: s.Problem.heroSide(tricks) >= s.Problem.Target,
	}
	if botted {
		s.dropBotSessionsLocked()
	}
}

// strayLocked applies a hero card no live line prescribes, on a problem
// whose author allows straying: the card stands, every other seat (and the
// rest of the hand) goes to the bots and the counted result is graded
// against the problem's Target, so an incorrect line simply fails (or
// succeeds, if the practitioner found something the author missed).
func (s *DrillSession) strayLocked(seat int, card string) error {
	s.Wrong++
	s.applyLocked(DrillPlay{Seat: seat, Card: card})
	s.Bot = true
	if len(s.Plays) == 52 {
		s.finishBotLocked()
		return nil
	}
	return s.advanceLocked()
}

func (s *DrillSession) finishClaimLocked(tricks int) {
	s.gradeLocked(tricks, true, false)
}

func (s *DrillSession) finishCountedLocked() {
	s.gradeLocked(s.DeclTricks, false, false)
}

// finishBotLocked grades a hand the bots finished against the target.
func (s *DrillSession) finishBotLocked() {
	s.gradeLocked(s.DeclTricks, false, true)
}

// dropBotSessionsLocked tells the sidecar to forget this session's bot
// card-play sessions.
func (s *DrillSession) dropBotSessionsLocked() {
	var out struct{}
	_ = botPost("/play/drop", map[string]any{"prefix": s.ID + ":"}, &out, botTimeout)
}

// endScriptLocked applies the ending of the line the script ran out on.
func (s *DrillSession) endScriptLocked() {
	l := s.liveLine()
	switch {
	case l.Bot != nil:
		s.Bot = true
	case l.Claim != nil:
		s.finishClaimLocked(l.Claim.Tricks)
	default:
		s.finishCountedLocked() // all 52 scripted cards were played
	}
}

// advanceLocked runs the session until the hero must act or the hand ends.
// During the scripted phase every non-hero seat plays the author's card
// instantly. Once the bots take over, exactly ONE bot card is played per
// call -- the sidecar's double-dummy choice is slow, so the client paces the
// bot cards one per request, like the play-a-hand table does.
func (s *DrillSession) advanceLocked() error {
	if s.Done {
		return nil
	}
	for !s.Bot {
		n := len(s.Plays)
		var active []int
		for _, li := range s.Live {
			if n < len(s.Problem.Lines[li].Plays) {
				active = append(active, li)
			}
		}
		if len(active) == 0 {
			s.endScriptLocked()
			break
		}
		s.Live = active
		seat := s.nextSeatLocked()
		if s.heroControls(seat) {
			return nil // wait for the hero
		}
		s.applyLocked(DrillPlay{Seat: seat, Card: s.Problem.Lines[active[0]].Plays[n].Card})
	}
	if s.Done || !s.Bot || len(s.Plays) == 52 {
		if s.Bot && len(s.Plays) == 52 {
			s.finishBotLocked()
		}
		return nil
	}
	seat := s.nextSeatLocked()
	if s.heroControls(seat) {
		return nil // the hero plays any legal card in the bot phase
	}
	card, err := s.botChooseLocked(seat)
	if err != nil {
		return err
	}
	if !slices.Contains(s.legalCardsLocked(seat), card) {
		return errors.New("sidecar returned an illegal card")
	}
	s.applyLocked(DrillPlay{Seat: seat, Card: card})
	if len(s.Plays) == 52 {
		s.finishBotLocked()
	}
	return nil
}

// botChooseLocked asks the sidecar for the card a bot seat should play, with
// the same request shape as the play-a-hand table (see play.go botChoose).
func (s *DrillSession) botChooseLocked(seat int) (string, error) {
	p := &s.Problem
	dummy := (s.contract.Declarer + 2) % 4
	var session string
	known := map[string]string{strconv.Itoa(dummy): p.Hands[dummy]}
	if seat == s.contract.Declarer || seat == dummy {
		session = s.ID + ":decl"
		known[strconv.Itoa(s.contract.Declarer)] = p.Hands[s.contract.Declarer]
	} else {
		session = s.ID + ":def" + strconv.Itoa(seat)
		known[strconv.Itoa(seat)] = p.Hands[seat]
	}
	plays := make([][2]any, 0, len(s.Plays))
	for _, dp := range s.Plays {
		plays = append(plays, [2]any{dp.Seat, dp.Card})
	}
	body := map[string]any{
		"session": session,
		"dealer":  p.Dealer,
		"vuln":    p.Vuln,
		"calls":   callsOrEmpty(p.Calls),
		"known":   known,
		"plays":   plays,
		"to_act":  seat,
	}
	var out struct {
		Seat int    `json:"seat"`
		Card string `json:"card"`
	}
	if err := botPost("/play/choose", body, &out, botPlayTimeout); err != nil {
		return "", err
	}
	card := normalizeCard(out.Card)
	if card == "" {
		return "", errors.New("sidecar returned an invalid card")
	}
	return card, nil
}

// stateLocked renders the client-visible session. During the play the hero
// sees their own hand only (plus the dummy once tabled); the script's
// messages stay hidden until the hand is over, and even a wrong pick reveals
// nothing -- the hint endpoint serves the current card's message on explicit
// request. The target travels from the very first state so the practitioner
// knows what to aim for before playing a card; how a line ends (claim, bots,
// played out) is never revealed up front. The problem's Explanation travels
// only with the done state.
func (s *DrillSession) stateLocked() map[string]any {
	p := &s.Problem
	st := map[string]any{
		"mode":           "drill",
		"id":             s.ID,
		"problemId":      p.ID,
		"title":          p.Title,
		"owner":          p.Owner,
		"role":           p.Role,
		"heroSeat":       p.HeroSeat,
		"difficulty":     p.Difficulty,
		"dealer":         p.Dealer,
		"vuln":           p.Vuln,
		"calls":          callsOrEmpty(p.Calls),
		"bidComment":     p.BidComment,
		"problemComment": p.ProblemComment,
		"contract":       s.contract,
		"declarer":       s.contract.Declarer,
		"dummy":          (s.contract.Declarer + 2) % 4,
		"opener":         (s.contract.Declarer + 1) % 4,
		"trump":          string(s.trump),
		"hand":           p.Hands[p.HeroSeat],
		"heroSeats":      p.heroSeats(),
		"lines":          len(p.Lines),
		"target":         p.Target,
		"allowWrong":     p.AllowWrong,
		"bot":            s.Bot,
		"wrong":          s.Wrong,
		"done":           s.Done,
	}
	plays := make([]map[string]any, 0, len(s.Plays))
	for _, dp := range s.Plays {
		plays = append(plays, map[string]any{"seat": dp.Seat, "card": dp.Card})
	}
	counts := [4]int{}
	for seat := 0; seat < 4; seat++ {
		counts[seat] = len(s.remainingHandLocked(seat))
	}
	dummyHand := ""
	if len(s.Plays) > 0 {
		dummyHand = p.Hands[(s.contract.Declarer+2)%4]
	}
	tricksDone := len(s.Plays) / 4
	st["plays"] = plays
	st["declTricks"] = s.DeclTricks
	st["defTricks"] = tricksDone - s.DeclTricks
	st["counts"] = counts
	st["dummyHand"] = dummyHand
	if s.Done {
		st["phase"] = "done"
		st["yourTurn"] = false
		st["nextSeat"] = -1
		st["hands"] = p.Hands
		st["result"] = s.Result
		st["explanation"] = p.Explanation
		// The review follows what ACTUALLY happened: the applied plays, with
		// the followed line's messages and equivalents on the scripted hero
		// cards (the bot-phase tail has no script).
		l := s.liveLine()
		line := make([]map[string]any, 0, len(s.Plays))
		for i, dp := range s.Plays {
			m := map[string]any{"seat": dp.Seat, "card": dp.Card}
			// the script's reasoning only fits cards it actually prescribed
			// (or an equivalent one); a strayed card carries no message
			if i < len(l.Plays) && s.heroControls(dp.Seat) && s.lineMatchesAt(l, i, dp.Seat, dp.Card) {
				msg := strings.TrimSpace(l.Plays[i].Msg)
				if msg == "" {
					msg = defaultHeroMsg(dp.Card)
				}
				m["msg"] = msg
				if alt := l.Plays[i].Alts; len(alt) > 0 {
					m["alts"] = alt
				}
			}
			line = append(line, m)
		}
		st["line"] = line
		st["scriptLen"] = len(l.Plays)
		return st
	}
	st["phase"] = "play"
	seat := s.nextSeatLocked()
	st["nextSeat"] = seat
	st["yourTurn"] = s.heroControls(seat)
	if st["yourTurn"].(bool) {
		st["actSeat"] = seat
		st["legal"] = s.legalCardsLocked(seat)
	}
	return st
}

// enrichDoneRating attaches the problem's live rating summary to a finished
// session state so the done view can show the average and the user's stars.
// Finishing a run also stamps the user's played flag on the problem.
// Called with s.mu held; takes drillsMu (the only nesting order used).
func enrichDoneRating(st map[string]any, username string) {
	if st["phase"] != "done" {
		return
	}
	pid, _ := st["problemId"].(string)
	drillsMu.Lock()
	if p := drills[pid]; p != nil {
		markPlayedLocked(p, username)
		avg, count := ratingSummaryLocked(p)
		st["rating"] = avg
		st["ratings"] = count
		st["yourRating"] = p.Ratings[username]
	}
	drillsMu.Unlock()
}

// markPlayedLocked stamps the first finished run of `username` on a problem.
// Callers must hold drillsMu.
func markPlayedLocked(p *DrillProblem, username string) {
	if username == "" {
		return
	}
	if _, ok := p.Played[username]; ok {
		return
	}
	if p.Played == nil {
		p.Played = map[string]string{}
	}
	p.Played[username] = time.Now().UTC().Format(time.RFC3339)
	if err := saveDrillLocked(p); err != nil {
		log.Printf("Could not persist played flag for drill %s: %v", p.ID, err)
	}
}

// writeSessionState renders a session (with the live rating summary attached
// once the hand is done). Called with s.mu held.
func writeSessionState(w http.ResponseWriter, s *DrillSession, username string) {
	st := s.stateLocked()
	enrichDoneRating(st, username)
	writeJSON(w, st)
}

func lookupDrillSession(id, username string) *DrillSession {
	drillSessMu.Lock()
	defer drillSessMu.Unlock()
	s := drillSess[id]
	if s == nil || s.User != username {
		return nil
	}
	return s
}

func gcDrillSessionsLocked() {
	now := time.Now()
	for id, s := range drillSess {
		if now.Sub(s.Created) > drillSessionTTL {
			delete(drillSess, id)
		}
	}
}

// errDrillBots marks a failure of the bot sidecar (502), as opposed to a
// problem-validation failure (400).
var errDrillBots = errors.New("bot service")

// deepCopy duplicates everything mutable so a session fully owns its
// snapshot; validation rebuilds the equivalence classes on the copy.
func (p *DrillProblem) deepCopy() DrillProblem {
	q := *p
	q.Calls = append([]string(nil), p.Calls...)
	q.Lines = make([]DrillLine, len(p.Lines))
	for i, l := range p.Lines {
		q.Lines[i].Plays = make([]DrillPlay, len(l.Plays))
		copy(q.Lines[i].Plays, l.Plays)
		for j := range q.Lines[i].Plays {
			if l.Plays[j].Alts != nil {
				q.Lines[i].Plays[j].Alts = append([]string(nil), l.Plays[j].Alts...)
			}
		}
		if l.Claim != nil {
			c := *l.Claim
			q.Lines[i].Claim = &c
		}
		if l.Bot != nil {
			b := *l.Bot
			q.Lines[i].Bot = &b
		}
	}
	return q
}

// newDrillSession snapshots a problem and runs the script up to the hero's
// first turn (which, for a bot line with no script left, may already mean a
// first sidecar call).
func newDrillSession(p *DrillProblem, username string) (*DrillSession, error) {
	snapshot := p.deepCopy()
	contract, err := validateDrill(&snapshot)
	if err != nil {
		return nil, err
	}
	s := &DrillSession{
		ID: generateID(), User: username, Problem: snapshot,
		contract: contract, trump: contract.Denom[0], Created: time.Now(),
	}
	s.Live = make([]int, len(snapshot.Lines))
	for i := range s.Live {
		s.Live[i] = i
	}
	if err := s.advanceLocked(); err != nil {
		return nil, fmt.Errorf("%w: %v", errDrillBots, err)
	}
	return s, nil
}

// ---- Handlers ----

// GET /api/drills - the library: metadata for every problem, no hands, no
// script, no comments (nothing that could spoil a practice run).
func handleDrillsList(w http.ResponseWriter, r *http.Request) {
	if requireLogin(w, r) == "" {
		return
	}
	drillsMu.Lock()
	me := currentUser(r)
	type info struct {
		ID         string  `json:"id"`
		Owner      string  `json:"owner"`
		Title      string  `json:"title"`
		Role       string  `json:"role"`
		HeroSeat   int     `json:"heroSeat"`
		Difficulty int     `json:"difficulty"`
		Dealer     int     `json:"dealer"`
		Vuln       int     `json:"vuln"`
		Contract   string  `json:"contract"`
		Lines      int     `json:"lines"`
		Rating     float64 `json:"rating"`
		Ratings    int     `json:"ratings"`
		YourRating int     `json:"yourRating"`
		Played     bool    `json:"played"`
		Created    string  `json:"created"`
		Updated    string  `json:"updated"`
	}
	list := make([]info, 0, len(drills))
	for _, p := range drills {
		label := "passed out"
		if c := contractFromCalls(p.Dealer, p.Calls); c != nil {
			label = c.String()
		}
		avg, count := ratingSummaryLocked(p)
		_, played := p.Played[me]
		list = append(list, info{
			ID: p.ID, Owner: p.Owner, Title: p.Title, Role: p.Role,
			HeroSeat: p.HeroSeat, Difficulty: p.Difficulty, Dealer: p.Dealer,
			Vuln: p.Vuln, Contract: label, Lines: len(p.Lines),
			Rating: avg, Ratings: count, YourRating: p.Ratings[me],
			Played: played, Created: p.Created, Updated: p.Updated,
		})
	}
	drillsMu.Unlock()
	// Highest rated first; unrated problems sink below rated ones, then the
	// most ratings wins ties, then the most recently updated.
	sort.Slice(list, func(i, j int) bool {
		if list[i].Rating != list[j].Rating {
			return list[i].Rating > list[j].Rating
		}
		if list[i].Ratings != list[j].Ratings {
			return list[i].Ratings > list[j].Ratings
		}
		return list[i].Updated > list[j].Updated
	})
	writeJSON(w, map[string]any{"problems": list})
}

// GET /api/drills/{id} - the full problem, for its owner's editor only.
func handleDrillGet(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	p := lookupDrill(id)
	if p == nil {
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	if p.Owner != username {
		jsonError(w, http.StatusForbidden, "only the author can open this problem")
		return
	}
	writeJSON(w, p)
}

// decodeDrillBody reads and validates a create/edit payload.
func decodeDrillBody(w http.ResponseWriter, r *http.Request) (*DrillProblem, *PlayContract, bool) {
	limitBody(w, r, maxBodyAuth)
	var p DrillProblem
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return nil, nil, false
	}
	contract, err := validateDrill(&p)
	if err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return nil, nil, false
	}
	return &p, contract, true
}

// POST /api/drills - create a problem owned by the current user.
func handleDrillCreate(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	p, _, ok := decodeDrillBody(w, r)
	if !ok {
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	p.ID = generateID()
	p.Owner = username
	p.Created = now
	p.Updated = now
	drillsMu.Lock()
	drills[p.ID] = p
	err := saveDrillLocked(p)
	drillsMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill %s: %v", p.ID, err)
		jsonError(w, http.StatusInternalServerError, "could not save the problem")
		return
	}
	writeJSON(w, map[string]string{"id": p.ID})
}

// PUT /api/drills/{id} - the author edits their problem.
func handleDrillPut(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	p, _, ok := decodeDrillBody(w, r)
	if !ok {
		return
	}
	drillsMu.Lock()
	existing := drills[id]
	if existing == nil {
		drillsMu.Unlock()
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	if existing.Owner != username {
		drillsMu.Unlock()
		jsonError(w, http.StatusForbidden, "only the author can edit this problem")
		return
	}
	p.ID = id
	p.Owner = existing.Owner
	p.Created = existing.Created
	p.Ratings = existing.Ratings // ratings and played stamps are server-side only
	p.Played = existing.Played
	p.Updated = time.Now().UTC().Format(time.RFC3339)
	drills[id] = p
	err := saveDrillLocked(p)
	drillsMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill %s: %v", id, err)
		jsonError(w, http.StatusInternalServerError, "could not save the problem")
		return
	}
	writeJSON(w, map[string]any{"ok": true, "id": id})
}

// DELETE /api/drills/{id} - the author removes their problem.
func handleDrillDelete(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	drillsMu.Lock()
	p := drills[id]
	if p == nil {
		drillsMu.Unlock()
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	if p.Owner != username {
		drillsMu.Unlock()
		jsonError(w, http.StatusForbidden, "only the author can delete this problem")
		return
	}
	delete(drills, id)
	drillsMu.Unlock()
	if err := os.Remove(drillPath(id)); err != nil && !os.IsNotExist(err) {
		log.Printf("Could not remove drill file: %v", err)
	}
	removeDrillFromArchives(id)
	writeJSON(w, map[string]bool{"ok": true})
}

// POST /api/drills/{id}/rate - rate a problem after finishing a practice run.
// Body: {stars: 1..5, sessionId}. The session must belong to the caller, be
// finished, and be a run of this problem; authors cannot rate their own.
func handleDrillRate(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Stars     int    `json:"stars"`
		SessionID string `json:"sessionId"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if body.Stars < 1 || body.Stars > 5 {
		jsonError(w, http.StatusBadRequest, "stars must be between 1 and 5")
		return
	}
	s := lookupDrillSession(body.SessionID, username)
	if s == nil {
		jsonError(w, http.StatusForbidden, "play the problem before rating it")
		return
	}
	s.mu.Lock()
	played := s.Done && s.Problem.ID == id
	s.mu.Unlock()
	if !played {
		jsonError(w, http.StatusForbidden, "finish the problem before rating it")
		return
	}
	drillsMu.Lock()
	p := drills[id]
	if p == nil {
		drillsMu.Unlock()
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	if p.Owner == username {
		drillsMu.Unlock()
		jsonError(w, http.StatusForbidden, "you cannot rate your own problem")
		return
	}
	if p.Ratings == nil {
		p.Ratings = map[string]int{}
	}
	p.Ratings[username] = body.Stars
	avg, count := ratingSummaryLocked(p)
	err := saveDrillLocked(p)
	drillsMu.Unlock()
	if err != nil {
		log.Printf("Could not persist drill %s: %v", id, err)
		jsonError(w, http.StatusInternalServerError, "could not save the rating")
		return
	}
	writeJSON(w, map[string]any{"rating": avg, "ratings": count, "yourRating": body.Stars})
}

// POST /api/drills/{id}/session - start practicing a problem.
func handleDrillSessionNew(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	id := r.PathValue("id")
	if !validID(id) {
		jsonError(w, http.StatusBadRequest, "invalid problem ID")
		return
	}
	p := lookupDrill(id)
	if p == nil {
		jsonError(w, http.StatusNotFound, "no such problem")
		return
	}
	drillsMu.Lock()
	snapshot := p.deepCopy()
	drillsMu.Unlock()
	s, err := newDrillSession(&snapshot, username)
	if err != nil {
		if errors.Is(err, errDrillBots) {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
		} else {
			jsonError(w, http.StatusBadRequest, err.Error())
		}
		return
	}
	drillSessMu.Lock()
	gcDrillSessionsLocked()
	drillSess[s.ID] = s
	drillSessMu.Unlock()
	s.mu.Lock()
	st := s.stateLocked()
	s.mu.Unlock()
	enrichDoneRating(st, username)
	writeJSON(w, st)
}

// GET /api/drill/session/{sid} - resume a practice session, and in the bot
// phase fetch the next single bot card (one per request, so the client can
// pace the reveals like at the play-a-hand table).
func handleDrillSessionGet(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	sid := r.PathValue("sid")
	if !validID(sid) {
		jsonError(w, http.StatusBadRequest, "invalid session ID")
		return
	}
	s := lookupDrillSession(sid, username)
	if s == nil {
		jsonError(w, http.StatusNotFound, "no such session")
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.Done && !s.heroControls(s.nextSeatLocked()) {
		if err := s.advanceLocked(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
	}
	writeSessionState(w, s, username)
}

// POST /api/drill/session/{sid}/card - the hero's pick. While the script is
// in charge only the authored card (or one of its declared equivalents) is
// accepted; anything else answers a bare 409 and plays nothing -- the
// author's message is NOT revealed (the client buzzes and flashes red; the
// hint stays behind its opt-in endpoint). Once the bots finish the hand, ANY
// legal card is accepted and the result is graded at the end.
func handleDrillSessionCard(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	sid := r.PathValue("sid")
	if !validID(sid) {
		jsonError(w, http.StatusBadRequest, "invalid session ID")
		return
	}
	limitBody(w, r, maxBodyAuth)
	var body struct {
		Card string `json:"card"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	card := normalizeCard(body.Card)
	if card == "" {
		jsonError(w, http.StatusBadRequest, "invalid card")
		return
	}
	s := lookupDrillSession(sid, username)
	if s == nil {
		jsonError(w, http.StatusNotFound, "no such session")
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Done {
		writeSessionState(w, s, username)
		return
	}
	seat := s.nextSeatLocked()
	if !s.heroControls(seat) {
		// a bot turn that stalled on a sidecar failure: advance one card
		if err := s.advanceLocked(); err != nil {
			jsonError(w, http.StatusBadGateway, botUnavailableMsg(err))
			return
		}
		writeSessionState(w, s, username)
		return
	}
	if !slices.Contains(s.legalCardsLocked(seat), card) {
		jsonError(w, http.StatusBadRequest, "illegal card: "+card)
		return
	}
	if s.Bot {
		// the bots finish the hand: the hero may play anything legal
		s.applyLocked(DrillPlay{Seat: seat, Card: card})
		if len(s.Plays) == 52 {
			s.finishBotLocked()
		} else if err := s.advanceLocked(); err != nil {
			// the hero's card stands; the client's polling GET retries the bot
			log.Printf("drill session %s: bot advance: %v", s.ID, err)
		}
		writeSessionState(w, s, username)
		return
	}
	var matched []int
	for _, li := range s.Live {
		if s.lineAccepts(li, seat, card) {
			matched = append(matched, li)
		}
	}
	if len(matched) == 0 {
		if !s.Problem.AllowWrong {
			s.Wrong++
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			json.NewEncoder(w).Encode(map[string]any{
				"error": "not the card the author played",
				"wrong": s.Wrong,
			})
			return
		}
		// the author lets the practitioner stray: the wrong card stands and
		// the bots finish the hand, graded against the problem's target
		if err := s.strayLocked(seat, card); err != nil {
			// the hero's card stands; the client's polling GET retries the bot
			log.Printf("drill session %s: bot advance: %v", s.ID, err)
		}
		writeSessionState(w, s, username)
		return
	}
	s.Live = matched
	s.applyLocked(DrillPlay{Seat: seat, Card: card})
	if err := s.advanceLocked(); err != nil {
		log.Printf("drill session %s: bot advance: %v", s.ID, err)
	}
	writeSessionState(w, s, username)
}

// GET /api/drill/session/{sid}/hint - the author's message for the card the
// script expects right now. This is the opt-in hint: the practitioner asks
// for it with a button while it is their turn in the scripted phase; a wrong
// pick never reveals it. Using a hint does not count as a wrong pick. While
// the bots play the rest of the hand there is no script left to hint from.
func handleDrillSessionHint(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	sid := r.PathValue("sid")
	if !validID(sid) {
		jsonError(w, http.StatusBadRequest, "invalid session ID")
		return
	}
	s := lookupDrillSession(sid, username)
	if s == nil {
		jsonError(w, http.StatusNotFound, "no such session")
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case s.Done:
		jsonError(w, http.StatusConflict, "the hand is over -- the review shows every message")
		return
	case s.Bot:
		jsonError(w, http.StatusConflict, "the bots are playing the rest -- every legal card is fine")
		return
	case !s.heroControls(s.nextSeatLocked()):
		jsonError(w, http.StatusConflict, "it is not your turn")
		return
	}
	l := s.liveLine()
	if len(s.Plays) >= len(l.Plays) {
		jsonError(w, http.StatusConflict, "no hint for this card")
		return
	}
	scripted := l.Plays[len(s.Plays)]
	msg := strings.TrimSpace(scripted.Msg)
	if msg == "" {
		msg = defaultHeroMsg(scripted.Card)
	}
	writeJSON(w, map[string]any{"hint": msg})
}

// DELETE /api/drill/session/{sid} - abandon a practice session.
func handleDrillSessionDelete(w http.ResponseWriter, r *http.Request) {
	username := requireLogin(w, r)
	if username == "" {
		return
	}
	sid := r.PathValue("sid")
	if !validID(sid) {
		jsonError(w, http.StatusBadRequest, "invalid session ID")
		return
	}
	s := lookupDrillSession(sid, username)
	if s == nil {
		jsonError(w, http.StatusNotFound, "no such session")
		return
	}
	drillSessMu.Lock()
	delete(drillSess, sid)
	drillSessMu.Unlock()
	writeJSON(w, map[string]bool{"ok": true})
}
