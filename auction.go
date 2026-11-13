package main

import "strings"

// ---- Structural auction reader (port of bidding_dt.conventions.auction) ----
//
// auctionReader is fed the calls of an auction in dealer order and classifies
// any candidate call: opening / response / rebid / overcall roles, jump
// distance, raises vs new suits vs cuebids, XYZ, fourth-suit GF, 2-over-1,
// invitational jump shifts, splinters, Smolen (the stayman state machine over
// 1NT-2C-2D / 2NT-3C-3D), RKCB asks and step responses, double meanings and
// the strong-2C structure. Classification is purely structural (no hands), a
// faithful port of the Python AuctionReader.classify/feed the convention rules
// train against -- system_file.go maps the result onto the sections of the
// bot's system file (models/system.txt) to render the website's bid alerts.
// Keep the two in lockstep with bidding-dt/src/bidding_dt/conventions/.
//
// Call ids match data/vocab.py: P=1, X=2, XX=3, bids 4..38 ordered by level
// with denoms C=0 D=1 H=2 S=3 N=4. Seats are absolute N=0 E=1 S=2 W=3,
// side = seat % 2 (0 = N-S).
//
// The game-force / passed-hand-strength bookkeeping of the Python reader is
// training-side only and deliberately not ported.

const (
	callPass = uint8(1)
	callX    = uint8(2)
	callXX   = uint8(3)
	firstBid = 4

	denomN = 4 // C=0 D=1 H=2 S=3 N=4
)

// bidID is the call id of a bid (level 1-7, denom 0C 1D 2H 3S 4N).
func bidID(level, denom int) int { return firstBid + (level-1)*5 + denom }

func bidLevelOf(call int) int { return (call-firstBid)/5 + 1 }

func bidDenomOf(call int) int { return (call - firstBid) % 5 }

func isMajor(denom int) bool { return denom == 2 || denom == 3 }

// vocabID maps a canonical call (P/X/XX/1C..7N) onto its bidding-dt token id.
func vocabID(call string) (uint8, bool) {
	switch call {
	case "P":
		return callPass, true
	case "X":
		return callX, true
	case "XX":
		return callXX, true
	}
	if len(call) == 2 && call[0] >= '1' && call[0] <= '7' {
		if d := strings.IndexByte("CDHSN", call[1]); d >= 0 {
			return 4 + (call[0]-'1')*5 + uint8(d), true
		}
	}
	return 0, false
}

// call kinds
const (
	kindBid = iota
	kindPass
	kindDouble
	kindRedouble
)

// callInfo is the structural description of one (candidate) call in its
// auction: the Go twin of conventions.auction.CallInfo (alert-relevant fields
// only).
type callInfo struct {
	seat   int
	kind   int
	level  int
	denom  int
	callID int
	jump   int // levels above the cheapest legal bid in this denom

	newSuit    bool
	raiseBid   bool // partner's suit, hero has not bid it
	ownRebid   bool // hero's own previously-bid suit
	cuebid     bool // opponents' last-bid suit
	firstCall  bool
	passedHand bool

	opening      bool // first bid of the auction
	responder    bool // partner opened; hero silent so far
	advancer     bool // partner overcalled; hero silent so far
	openerRebid  bool
	overcall     bool // hero's side silent, opponents have bid
	balancing    bool // 2+ passes since the last bid
	strong2c     bool // partner opened the strong 2C
	sideStrong2c bool
	uncontested  bool

	is2over1    bool
	isJumpShift bool // single jump, new suit: invitational in this system
	isStrongJS  bool // opener's jump-shift rebid
	splinter    bool
	smolen      bool
	fourthSuit  bool

	xyz           bool // responder's call in a live XYZ auction
	xyz2c         bool
	xyz2d         bool
	xyzRelay      bool // opener's bid after partner's XYZ 2C
	xyzAfterRelay bool // responder's call after XYZ 2C-2D

	rkcbAsk      bool
	rkcbResponse bool
	rkcbStep     int // 1..4 = 5C..5S
	rkcbTrump    int

	partnerLastSuit int // denom partner last bid (-1 none)
	sideLastSuit    int
	dblCtx          string // takeout|balancing_takeout|negative|penalty|nt_penalty
}

// XYZ state machine stages
const (
	xyzNone = iota
	xyzS1
	xyzS2
	xyzS3
	xyzRelayStage
	xyzAfterRelayStage
)

// stayman/Smolen state machine stages
const (
	staymanNone = iota
	staymanNT
	staymanAsked
	staymanDenied
)

// auctionReader replays an auction from the dealer; see the file header.
type auctionReader struct {
	dealer int
	n      int
	calls  []int // every call fed so far, dealer order (pattern-layer prefix)

	highest        int // highest bid call id (0 = none)
	highDenom      int
	passesSinceBid int

	called    [4]bool
	seatSuits [4]uint8 // bit per denom C,D,H,S
	seatNT    [4]bool

	sideSuitSeq   [2][]int // ordered seat<<3|denom suit bids per side
	sideLastDenom [2]int
	sideHasBid    [2]bool

	opener     int
	openLevel  int
	openDenom  int
	oppNonpass int // opponent bid/X/XX calls so far
	oppBids    int // opponent *bid* calls (2/1 exception)

	strong2cSide int

	// game-force bookkeeping (mirrors the Python reader's gf tracking, so
	// the pattern layer's `gf` clause sees the same state on both sides)
	gf             [2]bool
	s2cWaiting     bool
	s2cOpenerCalls int

	xyzStage  int
	xyzOpener int
	xyzFor    int
	xyz2cBy   int

	staymanStage  int
	staymanNT     int
	staymanOpener int
	staymanFor    int

	rkcbAsker        int
	rkcbTrumpLatched int
}

func newAuctionReader(dealer int) *auctionReader {
	r := &auctionReader{
		dealer:           ((dealer % 4) + 4) % 4,
		highDenom:        -1,
		opener:           -1,
		strong2cSide:     -1,
		xyzOpener:        -1,
		xyzFor:           -1,
		xyz2cBy:          -1,
		staymanOpener:    -1,
		staymanFor:       -1,
		rkcbAsker:        -1,
		rkcbTrumpLatched: -1,
	}
	r.sideLastDenom = [2]int{-1, -1}
	return r
}

func (r *auctionReader) nextSeat() int { return (r.dealer + r.n) % 4 }

func (r *auctionReader) seatBid(seat int) bool {
	return r.seatSuits[seat] != 0 || r.seatNT[seat]
}

// lastSuitOf returns the denom `seat` last bid for its side (-1 anyone).
func (r *auctionReader) lastSuitOf(side, seat int) int {
	seq := r.sideSuitSeq[side]
	for i := len(seq) - 1; i >= 0; i-- {
		s, d := seq[i]>>3, seq[i]&7
		if seat < 0 || s == seat {
			return d
		}
	}
	return -1
}

func (r *auctionReader) sideSuitSet(side int) uint8 {
	var m uint8
	for _, sd := range r.sideSuitSeq[side] {
		m |= 1 << uint(sd&7)
	}
	return m
}

// classify describes `call` made by `seat` without mutating any state, so the
// same method describes a candidate action and (inside feed) a real one.
func (r *auctionReader) classify(call, seat int) callInfo {
	side := seat % 2
	partner := (seat + 2) % 4
	info := callInfo{seat: seat, callID: call, denom: -1,
		rkcbTrump: -1, partnerLastSuit: -1, sideLastSuit: -1}
	info.firstCall = !r.called[seat]
	info.passedHand = r.called[seat] && !r.seatBid(seat)
	info.uncontested = r.oppNonpass == 0
	info.balancing = r.passesSinceBid >= 2
	info.sideStrong2c = r.strong2cSide == side
	info.partnerLastSuit = r.lastSuitOf(side, partner)
	info.sideLastSuit = r.lastSuitOf(side, -1)

	sideBid := r.sideHasBid[side]
	partnerBid := r.seatBid(partner)
	heroBid := r.seatBid(seat)

	switch call {
	case int(callPass):
		info.kind = kindPass
		if r.xyzStage == xyzAfterRelayStage && seat == r.xyz2cBy {
			info.xyzAfterRelay = true
		}
		return info
	case int(callX):
		info.kind = kindDouble
		info.dblCtx = r.doubleCtx(heroBid, partnerBid)
		return info
	case int(callXX):
		info.kind = kindRedouble
		return info
	}

	// -- bids --
	level, denom := bidLevelOf(call), bidDenomOf(call)
	info.level, info.denom = level, denom
	lmin := 1
	for bidID(lmin, denom) <= r.highest {
		lmin++
	}
	info.jump = level - lmin

	if denom < denomN {
		oppLastDenom := r.sideLastDenom[1-side]
		info.cuebid = denom == oppLastDenom &&
			r.seatSuits[seat]&(1<<uint(denom)) == 0 &&
			r.seatSuits[partner]&(1<<uint(denom)) == 0
		info.ownRebid = r.seatSuits[seat]&(1<<uint(denom)) != 0
		info.raiseBid = r.seatSuits[partner]&(1<<uint(denom)) != 0 && !info.ownRebid
		info.newSuit = !info.ownRebid && !info.raiseBid
	}

	// 4NT with a suit bid by our side is RKCB in that suit
	if denom == denomN && level == 4 && info.sideLastSuit >= 0 {
		info.rkcbAsk = true
		info.rkcbTrump = info.sideLastSuit
	}

	if r.highest == 0 {
		info.opening = true
		return info
	}

	switch {
	case !sideBid:
		info.overcall = true // hero's side has not bid yet
	case !r.called[seat] && partner == r.opener:
		info.responder = true
	case !r.called[seat] && partnerBid:
		info.advancer = true
	case seat == r.opener:
		info.openerRebid = true
	}
	info.strong2c = r.strong2cSide == side && info.responder

	// XYZ (uncontested 1x-1y-1z): responder's 2C/2D are artificial and
	// override the natural reading, so this runs before the early returns.
	if r.xyzStage == xyzS3 && seat == r.xyzFor {
		info.xyz = true
		if call == bidID(2, 0) {
			info.xyz2c = true
			return info
		}
		if call == bidID(2, 1) {
			info.xyz2d = true
			return info
		}
	} else if r.xyzStage == xyzRelayStage && seat == r.xyzOpener {
		info.xyzRelay = true
	} else if r.xyzStage == xyzAfterRelayStage && seat == r.xyz2cBy {
		info.xyzAfterRelay = true
	}

	// Smolen: after a 1NT/2NT opening, stayman and the 2D/3D denial,
	// responder's 3M shows 4 in the bid major + 5 in the other, GF.
	if r.staymanStage == staymanDenied && seat == r.staymanFor &&
		level == 3 && isMajor(denom) {
		info.smolen = true
		return info
	}

	// RKCB step response: the asker's partner's next 5-level suit bid.
	if r.rkcbAsker >= 0 && seat == (r.rkcbAsker+2)%4 &&
		level == 5 && denom < denomN {
		info.rkcbResponse = true
		info.rkcbStep = denom + 1
		info.rkcbTrump = r.rkcbTrumpLatched
		return info
	}

	if denom == denomN || info.cuebid || info.ownRebid || info.raiseBid {
		return info // NT / cuebid / rebid / raise: no shift logic
	}

	// 2/1 game force: 2-level new-suit response to a 1-level opening by a
	// silent responder; a level-2 single "jump" only stays a 2/1 when the
	// opponents have not *bid* (doubles do not consume a level).
	if info.responder && info.firstCall && level == 2 &&
		r.openLevel == 1 && r.openDenom < denomN &&
		(info.jump == 0 || (info.jump == 1 && r.oppBids == 0)) {
		info.is2over1 = true
		return info
	}

	// fourth suit forcing (GF): the side has bid exactly 3 distinct suits,
	// hero bids the missing one, has bid before, uncontested
	set := r.sideSuitSet(side)
	if info.newSuit && info.uncontested && r.seatBid(seat) &&
		popCount(set) == 3 && set&(1<<uint(denom)) == 0 &&
		r.xyzStage == xyzNone && !info.sideStrong2c {
		info.fourthSuit = true
	}

	if !info.newSuit || info.sideStrong2c {
		return info
	}

	// single jump shift (level 2-3): invitational -- also in competition;
	// opener's rebid is the standard strong one instead. In a live XYZ
	// auction a direct level-2 jump is natural (invites go through 2C).
	if info.jump == 1 && level >= 2 && level <= 3 && !info.passedHand &&
		!(info.xyz && level == 2) {
		switch {
		case info.openerRebid:
			info.isStrongJS = true
		case sideBid || info.overcall:
			info.isJumpShift = true
		}
	}

	// double jump shift below 4H: splinter (shortness + fit for partner)
	if info.jump == 2 && call < bidID(4, 2) &&
		info.partnerLastSuit >= 0 && !info.passedHand {
		info.splinter = true
	}
	return info
}

func popCount(m uint8) int {
	n := 0
	for m != 0 {
		n += int(m & 1)
		m >>= 1
	}
	return n
}

func (r *auctionReader) doubleCtx(heroBid, partnerBid bool) string {
	if r.highDenom == denomN {
		if heroBid || partnerBid {
			return "penalty"
		}
		return "nt_penalty"
	}
	if heroBid {
		return "penalty"
	}
	if partnerBid {
		return "negative"
	}
	if r.passesSinceBid >= 2 {
		return "balancing_takeout"
	}
	return "takeout"
}

// feed classifies the next call, then folds it into the auction state.
func (r *auctionReader) feed(call int) callInfo {
	seat := r.nextSeat()
	side := seat % 2
	info := r.classify(call, seat)

	r.called[seat] = true
	switch call {
	case int(callPass):
		r.passesSinceBid++
	default:
		level, denom := info.level, info.denom
		r.highest = call
		r.highDenom = denom
		r.passesSinceBid = 0
		if denom == denomN {
			r.seatNT[seat] = true
		} else {
			r.seatSuits[seat] |= 1 << uint(denom)
			r.sideSuitSeq[side] = append(r.sideSuitSeq[side], seat<<3|denom)
			r.sideLastDenom[side] = denom
		}
		r.sideHasBid[side] = true
		if r.opener < 0 {
			r.opener = seat
			r.openLevel, r.openDenom = level, denom
			if level == 2 && denom == 0 { // strong 2C opening
				r.strong2cSide = side
			}
		}
	}

	// opponent activity (uncontested flag / 2-over-1 jump exception); must
	// run before the XYZ/stayman updates, exactly as in the Python reader
	if r.opener >= 0 && side != r.opener%2 && call != int(callPass) {
		r.oppNonpass++
		if call >= firstBid {
			r.oppBids++
		}
	}

	// strong 2C: responder's 2D is the waiting bid (it makes opener's 2NT
	// rebid invitational instead of a game force)
	if r.strong2cSide == side && seat == (r.opener+2)%4 &&
		info.firstCall && call == bidID(2, 1) {
		r.s2cWaiting = true
	}

	r.updateGF(seat, side, info)
	r.updateXYZ(call, seat, info)
	r.updateStayman(call, seat, info)
	r.updateRKCB(call, seat, info)
	r.calls = append(r.calls, call)
	r.n++
	return info
}

// updateGF tracks the per-side game force, mirroring AuctionReader._update_gf.
func (r *auctionReader) updateGF(seat, side int, info callInfo) {
	if info.kind != kindBid {
		return
	}
	// strong 2C: after a 2D waiting response, opener's 2NT rebid is
	// invitational (22-24), not a game force
	if r.strong2cSide == side && seat == r.opener {
		r.s2cOpenerCalls++
		if r.s2cOpenerCalls == 2 && r.s2cWaiting &&
			info.level == 2 && info.denom == denomN {
			r.gf[side] = false
		}
	}
	if info.is2over1 || info.fourthSuit || info.xyz2d || info.smolen {
		r.gf[side] = true
	}
	if isGameOrAbove(info.level, info.denom) {
		r.gf[side] = false // obligation fulfilled
	}
}

func (r *auctionReader) updateXYZ(call, seat int, info callInfo) {
	stage := r.xyzStage
	if stage == xyzNone {
		if info.opening && info.level == 1 && info.denom < denomN {
			r.xyzStage = xyzS1
			r.xyzOpener = seat
		}
		return
	}
	// any opponent non-pass call kills basic XYZ
	if r.oppNonpass > 0 {
		r.xyzStage = xyzNone
		return
	}
	switch stage {
	case xyzS1:
		if info.kind == kindPass {
			return // waiting for the responder
		}
		if seat == (r.xyzOpener+2)%4 && info.level == 1 && info.denom < denomN {
			r.xyzStage = xyzS2
		} else {
			r.xyzStage = xyzNone
		}
	case xyzS2:
		if info.kind == kindPass {
			return // waiting for opener's rebid
		}
		if seat == r.xyzOpener && info.level == 1 && info.denom < denomN {
			r.xyzStage = xyzS3
			r.xyzFor = (r.xyzOpener + 2) % 4
		} else {
			r.xyzStage = xyzNone
		}
	case xyzS3:
		if seat != r.xyzFor {
			if info.kind != kindPass {
				r.xyzStage = xyzNone
			}
			return
		}
		if info.xyz2c {
			r.xyzStage = xyzRelayStage
			r.xyz2cBy = seat
		} else {
			r.xyzStage = xyzNone // bypassed, or GF 2D
		}
	case xyzRelayStage:
		if seat != r.xyzOpener {
			return // passes wait for the relay
		}
		if call == bidID(2, 1) {
			r.xyzStage = xyzAfterRelayStage
		} else {
			r.xyzStage = xyzNone
		}
	case xyzAfterRelayStage:
		if seat == r.xyz2cBy {
			r.xyzStage = xyzNone
		}
	}
}

func (r *auctionReader) updateStayman(call, seat int, info callInfo) {
	stage := r.staymanStage
	if stage == staymanNone {
		if info.opening && info.denom == denomN &&
			(info.level == 1 || info.level == 2) {
			r.staymanStage = staymanNT
			r.staymanNT = info.level
			r.staymanOpener = seat
		}
		return
	}
	if r.oppNonpass > 0 { // interference kills stayman
		r.staymanStage = staymanNone
		return
	}
	if info.kind == kindPass {
		return
	}
	ask := bidID(r.staymanNT+1, 0)  // 2C over 1NT, 3C over 2NT
	deny := bidID(r.staymanNT+1, 1) // the matching 2D/3D denial
	switch stage {
	case staymanNT:
		if seat == (r.staymanOpener+2)%4 && call == ask {
			r.staymanStage = staymanAsked
			r.staymanFor = seat
		} else {
			r.staymanStage = staymanNone
		}
	case staymanAsked:
		if seat == r.staymanOpener && call == deny {
			r.staymanStage = staymanDenied
		} else {
			r.staymanStage = staymanNone
		}
	case staymanDenied:
		r.staymanStage = staymanNone // responder called (Smolen or not)
	}
}

func (r *auctionReader) updateRKCB(call, seat int, info callInfo) {
	// latch the ask; the asker's partner's next call consumes it. An
	// intervening opponent bid/double cancels the step-response window;
	// passes keep it open.
	if r.rkcbAsker >= 0 {
		partner := (r.rkcbAsker + 2) % 4
		if seat == partner || call != int(callPass) {
			r.rkcbAsker, r.rkcbTrumpLatched = -1, -1
		}
	}
	if info.rkcbAsk {
		r.rkcbAsker, r.rkcbTrumpLatched = seat, info.rkcbTrump
	}
}
