package main

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ---- Claim verification ----
//
// A player may claim tricks when it is their turn to play (or their dummy's):
// the declarer claims for declarer+dummy, a defender for themselves alone. The
// claim states how many of the remaining tricks the claiming side takes --
// any number from 0 to all of them -- and the unclaimed rest is conceded to
// the opponents. A claim of 0 (a concede) needs no proof and is accepted as
// is; any other claim is accepted only when it cannot fail -- when the
// claiming side wins the next `want` tricks IN A ROW (taking every trick it
// claims before losing any) in every legal layout of the unseen cards. A
// layout is legal when it agrees with everything the claimer can know: their
// own (and for the declarer the dummy's) cards, every card played so far,
// each seat's card count, and the voids proven by failing to follow suit.
// Any other distribution of the unseen cards must be assumed possible.
//
// The search models the hidden seats as drawing adversarially from a shared
// pool of unseen cards: on each turn a hidden seat either follows with any
// plausible card of the led suit, or shows out (which marks it permanently
// void in that suit) and plays anything else. Every real layout is one of the
// branches explored, so a claim accepted here also succeeds on the actual
// deal -- and positions too large to exhaust within claimVerifyBudget are
// rejected. Rejecting a correct claim is acceptable; accepting a wrong one
// is not.
//
// Unseen cards are only ever compared against cards whose rank is visible
// (the known hands and the cards already played), so ranks that sit between
// two consecutive visible ranks of a suit are interchangeable: the pool only
// counts cards per such "gap". Two pool cards of one gap meeting in a trick
// tie, and a tie always means a hidden seat won the trick -- the claim side
// lost it -- which is the conservative verdict.

const (
	claimVerifyBudget = time.Second // wall-clock cap for one claim search
	claimVerifyNodes  = 2_000_000   // node cap for one claim search
	claimMemoMax      = 1 << 20     // memo entries cap
)

// errClaimTimeout marks a search that exhausted its budget: the claim could
// not be verified and is refused.
var errClaimTimeout = errors.New("too complex to verify")

// suitNum maps a suit char (S,H,D,C -- the card-token order) to 0..3.
func suitNum(ch byte) int8 { return int8(strings.IndexByte("SHDC", ch)) }

// claimPlay is one card of the trick in progress (or of the partial trick the
// claim was made on). value is the card's trick-taking strength inside its
// suit, lower wins: an exactly known card (from a known hand, or revealed by
// being played) carries 2*rankIdx+2; a card drawn from a pool gap carries the
// gap's value -- 2*anchor+1 for the gap above every visible rank of the suit,
// 2*anchor+3 for the gap just below the visible rank `anchor`. Gap values sit
// strictly between the visible ranks they separate, so every comparison
// against a visible card is exact, and equal gap values only ever tie between
// two hidden seats.
type claimPlay struct {
	seat  int8
	suit  int8
	value int16
}

type claimSolver struct {
	trumpI   int8         // trump suit 0..3, or -1 for notrump
	claim    [4]bool      // the seats that must win every remaining trick
	kind     [4]int8      // 0 unseen, 1 known claim side, 2 known opponent
	hand     [4][4][]int8 // known seat: rankIdx per suit, ascending (strongest first)
	mask     [4][4]uint16 // which cards of hand[seat][suit] are still held
	gapVal   [4][]int16   // per suit: strength value of each pool gap, strongest first
	pool     [4][]int8    // per suit: unseen card count per gap
	count    [4]int8      // unseen seat: cards remaining
	voids    [4]uint8     // unseen seat: one bit per suit proven void
	trick    [4]claimPlay // the trick in progress
	tn       int8         // cards played to the current trick
	next     int8         // seat to act
	rem      int          // total cards still held (hands + pool)
	need     int          // claimed tricks the claim side still has to win
	memo     map[string]bool
	nodes    int
	deadline time.Time
}

// buildClaimSolver sets up the verification position. claim marks the seats
// the claim is made for, known the seats whose remaining cards the claimer
// sees (the claim side, plus the dummy once tabled); every other seat is
// unseen and shares the pool. next is the seat to act -- the claim may only
// be made by a claim-side seat on its turn.
func buildClaimSolver(hands [4]string, plays []PlayCard, trump byte, claim, known [4]bool, next int) (*claimSolver, error) {
	if next < 0 || next > 3 || !claim[next] || !known[next] {
		return nil, errors.New("a claim must be made by a claim-side player on their turn")
	}
	for seat := 0; seat < 4; seat++ {
		if claim[seat] && !known[seat] {
			return nil, errors.New("the claimer must see every hand of the claim side")
		}
	}
	sv := &claimSolver{
		trumpI:   -1,
		claim:    claim,
		next:     int8(next),
		memo:     map[string]bool{},
		deadline: time.Now().Add(claimVerifyBudget),
	}
	if trump != 'N' {
		t := suitNum(trump)
		if t < 0 {
			return nil, errors.New("bad trump suit")
		}
		sv.trumpI = t
	}
	played := make(map[string]bool, len(plays))
	for _, pc := range plays {
		if normalizeCard(pc.Card) != pc.Card || !validSeat(pc.Seat) {
			return nil, errors.New("bad play in the claim position")
		}
		played[pc.Card] = true
	}
	// what each seat has left; for the unseen seats only the COUNT is ever
	// used (card counts are public, the cards themselves are not)
	rem := [4][]string{}
	for seat := 0; seat < 4; seat++ {
		for _, c := range handCards(hands[seat]) {
			if !played[c] {
				rem[seat] = append(rem[seat], c)
			}
		}
	}
	for seat := 0; seat < 4; seat++ {
		switch {
		case claim[seat]:
			sv.kind[seat] = 1
		case known[seat]:
			sv.kind[seat] = 2
		default:
			sv.count[seat] = int8(len(rem[seat]))
			sv.rem += len(rem[seat])
		}
	}
	// voids proven by the play: whoever failed to follow suit holds none of it
	for i, pc := range plays {
		if i%4 == 0 {
			continue
		}
		led := cardSuit(plays[(i/4)*4].Card)
		if cardSuit(pc.Card) != led {
			sv.voids[pc.Seat] |= 1 << suitNum(led)
		}
	}
	// Boundary ranks per suit: every card whose exact rank is visible -- the
	// known hands' remaining cards and every card already played. The pool's
	// gaps are the intervals between consecutive boundaries.
	bounds := [4][]int8{}
	seen := make(map[string]bool, 52)
	for seat := 0; seat < 4; seat++ {
		if sv.kind[seat] == 0 {
			continue
		}
		for _, c := range rem[seat] {
			s := suitNum(cardSuit(c))
			bounds[s] = append(bounds[s], int8(cardRankIdx(c)))
			seen[c] = true
		}
	}
	for _, pc := range plays {
		s := suitNum(cardSuit(pc.Card))
		bounds[s] = append(bounds[s], int8(cardRankIdx(pc.Card)))
	}
	for s := 0; s < 4; s++ {
		sort.Slice(bounds[s], func(i, j int) bool { return bounds[s][i] < bounds[s][j] })
		bs := bounds[s]
		gv := make([]int16, len(bs)+1)
		if len(bs) == 0 {
			gv[0] = 0 // nothing of the suit is visible: one gap holding it all
		} else {
			gv[0] = int16(2*bs[0] + 1)
			for j := range bs {
				gv[j+1] = int16(2*bs[j] + 3)
			}
		}
		sv.gapVal[s] = gv
		sv.pool[s] = make([]int8, len(gv))
	}
	// the known hands, and the pool of everything else
	for seat := 0; seat < 4; seat++ {
		if sv.kind[seat] == 0 {
			continue
		}
		for _, c := range rem[seat] {
			s := suitNum(cardSuit(c))
			sv.hand[seat][s] = append(sv.hand[seat][s], int8(cardRankIdx(c)))
		}
		for s := 0; s < 4; s++ {
			h := sv.hand[seat][s]
			sort.Slice(h, func(i, j int) bool { return h[i] < h[j] })
			sv.mask[seat][s] = uint16(1<<len(h)) - 1
			sv.rem += len(h)
		}
	}
	poolTotal := 0
	for s := 0; s < 4; s++ {
		for r := 0; r < 13; r++ {
			c := string(cardRanks[r]) + string("SHDC"[s])
			if played[c] || seen[c] {
				continue
			}
			g := 0
			for _, b := range bounds[s] {
				if b >= int8(r) {
					break
				}
				g++
			}
			sv.pool[s][g]++
			poolTotal++
		}
	}
	unseen := 0
	for seat := 0; seat < 4; seat++ {
		unseen += int(sv.count[seat])
	}
	if poolTotal != unseen {
		return nil, errors.New("inconsistent claim position")
	}
	// the partial trick the claim was made on: every card on it is revealed
	tn := len(plays) % 4
	for i := 0; i < tn; i++ {
		pc := plays[len(plays)-tn+i]
		sv.trick[i] = claimPlay{
			seat:  int8(pc.Seat),
			suit:  suitNum(cardSuit(pc.Card)),
			value: int16(2*cardRankIdx(pc.Card) + 2),
		}
	}
	sv.tn = int8(tn)
	return sv, nil
}

// claimRemainingTricks is the number of tricks still to be decided (the trick
// in progress counts as one) once len(plays) cards are on the table.
func claimRemainingTricks(plays int) int { return (52 - plays + 3) / 4 }

// verifyClaim reports whether the claim side takes the next `want` tricks in
// a row -- every trick it claims, before losing any -- in every legal layout
// of the unseen cards; the tricks it does not claim are conceded and never
// looked at. A want of 0 is a concede: giving tricks away needs no proof and
// is accepted as is. (true, nil) accepts the claim; (false, nil) means the
// defense has a way to take a trick before the claim is fulfilled; (false,
// errClaimTimeout) means the position could not be settled in time -- which
// also refuses the claim.
func verifyClaim(hands [4]string, plays []PlayCard, trump byte, claim, known [4]bool, next, want int) (bool, error) {
	if want < 0 {
		return false, errors.New("a claim cannot take fewer than zero tricks")
	}
	if want == 0 {
		return true, nil // a concede: auto-accepted
	}
	sv, err := buildClaimSolver(hands, plays, trump, claim, known, next)
	if err != nil {
		return false, err
	}
	remaining := (sv.rem + 3) / 4
	if sv.rem == 0 {
		return false, errors.New("there are no tricks left to claim")
	}
	if want > remaining {
		return false, fmt.Errorf("cannot claim %d of the %d remaining tricks", want, remaining)
	}
	sv.need = want
	if sv.fastSolidRun() {
		// the fast path proves the claim side takes EVERY remaining trick,
		// which is more than any claim asks for
		return true, nil
	}
	return sv.search()
}

// ---- The search ----

func (sv *claimSolver) search() (bool, error) {
	sv.nodes++
	if sv.nodes%1024 == 0 && (sv.nodes > claimVerifyNodes || time.Now().After(sv.deadline)) {
		return false, errClaimTimeout
	}
	if sv.tn == 4 {
		w, ok := sv.trickWon()
		if !ok {
			return false, nil // the defense took a trick before the claim was fulfilled
		}
		sv.need--
		if sv.need == 0 {
			sv.need = 1
			return true, nil // every claimed trick is sure; the rest is conceded
		}
		if sv.rem == 0 {
			sv.need++
			return false, nil // the cards ran out before the claim did
		}
		// the next trick overwrites sv.trick from index 0: save the completed
		// one so the caller's other alternatives still see their own cards
		savedTrick, savedNext := sv.trick, sv.next
		sv.tn, sv.next = 0, int8(w)
		res, err := sv.search()
		sv.tn, sv.next, sv.trick = 4, savedNext, savedTrick
		sv.need++
		return res, err
	}
	key := sv.key()
	if v, ok := sv.memo[key]; ok {
		return v, nil
	}
	res, err := sv.turn()
	if err == nil && len(sv.memo) < claimMemoMax {
		sv.memo[key] = res
	}
	return res, err
}

// trickWon resolves the completed trick. It reports the winning seat and
// whether the claim side won: a value tie can only happen between two pool
// cards of one gap (both hidden seats), so a tie always loses the claim.
func (sv *claimSolver) trickWon() (int, bool) {
	suit := sv.trick[0].suit
	if sv.trumpI >= 0 {
		for i := 0; i < 4; i++ {
			if sv.trick[i].suit == sv.trumpI {
				suit = sv.trumpI
				break
			}
		}
	}
	best, win, tie := int16(1<<14), -1, false
	for i := 0; i < 4; i++ {
		c := sv.trick[i]
		if c.suit != suit {
			continue
		}
		switch {
		case c.value < best:
			best, win, tie = c.value, int(c.seat), false
		case c.value == best:
			tie = true
		}
	}
	if tie || win < 0 || !sv.claim[win] {
		return 0, false
	}
	return win, true
}

// key packs the mutable search state for the memo. The gap values, the claim
// side and the kinds never change, so they are not part of it.
func (sv *claimSolver) key() string {
	var b [112]byte
	n := 0
	for s := 0; s < 4; s++ {
		for _, c := range sv.pool[s] {
			b[n] = byte(c)
			n++
		}
	}
	for seat := 0; seat < 4; seat++ {
		if sv.kind[seat] == 0 {
			b[n], b[n+1] = byte(sv.count[seat]), sv.voids[seat]
			n += 2
			continue
		}
		for s := 0; s < 4; s++ {
			m := sv.mask[seat][s]
			b[n], b[n+1] = byte(m), byte(m>>8)
			n += 2
		}
	}
	b[n], b[n+1] = byte(sv.next), byte(sv.tn)
	n += 2
	for i := 0; i < int(sv.tn); i++ {
		b[n] = byte(sv.trick[i].seat)
		b[n+1] = byte(sv.trick[i].suit)
		b[n+2] = byte(sv.trick[i].value)
		n += 3
	}
	if n > len(b) {
		panic("claim key overflow")
	}
	return string(b[:n])
}

func (sv *claimSolver) turn() (bool, error) {
	seat := int(sv.next)
	if sv.kind[seat] == 0 {
		return sv.unseenTurn(seat)
	}
	return sv.knownTurn(seat)
}

// knownTurn plays a seat whose exact cards are visible. A claim-side seat
// picks the line that wins (any branch suffices); a known opponent -- the
// tabled dummy on a defender's claim -- plays adversarially (every branch
// must still win).
func (sv *claimSolver) knownTurn(seat int) (bool, error) {
	mine := sv.kind[seat] == 1
	led := int8(-1)
	if sv.tn > 0 {
		led = sv.trick[0].suit
	}
	force := led >= 0 && sv.mask[seat][led] != 0
	for s := 0; s < 4; s++ {
		if force && int8(s) != led {
			continue
		}
		for i := range sv.hand[seat][s] {
			bit := uint16(1) << i
			if sv.mask[seat][s]&bit == 0 {
				continue
			}
			sv.mask[seat][s] &^= bit
			sv.rem--
			sv.trick[sv.tn] = claimPlay{int8(seat), int8(s), int16(2*sv.hand[seat][s][i] + 2)}
			sv.tn++
			sv.next = int8((seat + 1) % 4)
			res, err := sv.search()
			sv.next = int8(seat)
			sv.tn--
			sv.rem++
			sv.mask[seat][s] |= bit
			if err != nil {
				return false, err
			}
			if mine && res {
				return true, nil
			}
			if !mine && !res {
				return false, nil
			}
		}
	}
	return !mine, nil
}

// unseenTurn plays a hidden seat: the adversary picks ANY card the pool can
// still supply -- following the led suit, or showing out of it (which marks
// the seat void forever) and playing another suit. The claim must survive
// every choice. Positions where no choice is left are unreachable by any real
// layout (card counts or voids contradict them), so they count as won.
func (sv *claimSolver) unseenTurn(seat int) (bool, error) {
	led := int8(-1)
	if sv.tn > 0 {
		led = sv.trick[0].suit
	}
	type opt struct {
		suit int8
		gap  int
		mark bool // showing out of the led suit
	}
	var opts []opt
	add := func(s int8, mark bool) {
		for g := range sv.pool[s] {
			if sv.pool[s][g] > 0 {
				opts = append(opts, opt{s, g, mark})
			}
		}
	}
	switch {
	case led < 0: // on lead: any suit not proven void
		for s := int8(0); s < 4; s++ {
			if sv.voids[seat]>>s&1 == 0 {
				add(s, false)
			}
		}
	case sv.voids[seat]>>led&1 != 0: // already void: anything else
		for s := int8(0); s < 4; s++ {
			if s != led && sv.voids[seat]>>s&1 == 0 {
				add(s, false)
			}
		}
	default: // follow the led suit, or show out of it and discard
		add(led, false)
		for s := int8(0); s < 4; s++ {
			if s != led && sv.voids[seat]>>s&1 == 0 {
				add(s, true)
			}
		}
	}
	if len(opts) == 0 {
		return true, nil
	}
	for _, o := range opts {
		sv.pool[o.suit][o.gap]--
		sv.count[seat]--
		sv.rem--
		oldVoids := sv.voids[seat]
		if o.mark {
			sv.voids[seat] |= 1 << led
		}
		sv.trick[sv.tn] = claimPlay{int8(seat), o.suit, sv.gapVal[o.suit][o.gap]}
		sv.tn++
		sv.next = int8((seat + 1) % 4)
		res, err := sv.search()
		sv.next = int8(seat)
		sv.tn--
		sv.rem++
		sv.voids[seat] = oldVoids
		sv.count[seat]++
		sv.pool[o.suit][o.gap]++
		if err != nil {
			return false, err
		}
		if !res {
			return false, nil
		}
	}
	return true, nil
}

// ---- Fast path for the classic "the rest are mine" ----

// fastSolidRun is a cheap sufficient check for the classic declarer claim: at
// a trick boundary, ONE claim-side hand -- the one on lead -- can run every
// remaining trick by itself. It leads all its trumps first, each beating any
// trump the defense can hold (which must all fall during the run) and any
// trump partner must follow with (partner must run out too, or it would ruff
// and take the lead), then its side cards strongest first. Partner's cards
// are paired strongest against strongest -- partner cooperates; the defense
// keeps its strongest cards as long as it can, so the runner's i-th weakest
// card of a suit must still beat the defense's i-th strongest. When this
// holds the claim is accepted without a search.
func (sv *claimSolver) fastSolidRun() bool {
	if sv.tn != 0 {
		return false
	}
	lead := int(sv.next)
	partner := -1
	for seat := 0; seat < 4; seat++ {
		if !sv.claim[seat] {
			continue
		}
		if sv.kind[seat] != 1 {
			return false
		}
		if seat != lead {
			if partner >= 0 {
				return false
			}
			partner = seat
		}
	}
	if partner < 0 {
		return false // a lone defender's claim: the search decides
	}
	mine := func(seat, s int) []int16 { // strength values, strongest first
		out := make([]int16, 0, len(sv.hand[seat][s]))
		for i := range sv.hand[seat][s] {
			if sv.mask[seat][s]>>uint(i)&1 == 1 {
				out = append(out, int16(2*sv.hand[seat][s][i]+2))
			}
		}
		return out
	}
	theirs := func(s int) []int16 { // the defense's pool, strongest first
		var out []int16
		for g, n := range sv.pool[s] {
			for k := int8(0); k < n; k++ {
				out = append(out, sv.gapVal[s][g])
			}
		}
		return out
	}
	check := func(s int, isTrump bool) bool {
		l, q, p := mine(lead, s), mine(partner, s), theirs(s)
		if isTrump && (len(p) > len(l) || len(q) > len(l)) {
			return false // trumps outlive the run: someone else ruffs or leads
		}
		for i := 0; i < len(q) && i < len(l); i++ {
			if l[i] >= q[i] {
				return false // partner would win a trick and take the lead
			}
		}
		for i := 1; i <= len(p) && i <= len(l); i++ {
			if l[len(l)-i] >= p[i-1] {
				return false // the defense keeps a card that beats the run
			}
		}
		return true
	}
	if sv.trumpI >= 0 && !check(int(sv.trumpI), true) {
		return false
	}
	for s := 0; s < 4; s++ {
		if int8(s) == sv.trumpI {
			continue
		}
		if !check(s, false) {
			return false
		}
	}
	return true
}
