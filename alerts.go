package main

// ---- Bid alerts from the bot's system file ----
//
// The bots' conventions are written in models/system.txt (SYSTEM_FILE
// overrides) -- the same text file bidding-dt interprets at the start of
// training, so what the bots are trained to play and what the site explains
// always agree. Alerts are generated from that file, not from self-play
// statistics: for every bot call the auction reader (auction.go) classifies
// the call in its sequence, system_file.go resolves it to the matching
// section of the system file, and the section's alert text is served to the
// client. A section flagged `artificial = yes` additionally carries the
// alert marker (XYZ, RKCB ask/response, splinter, smolen, fourth-suit GF,
// support cuebid, weak two, strong 2C, ...); natural calls covered by the
// file (openings, responses, raises, rebids, 2/1, overcalls, doubles) show
// their meaning as plain text, and calls the file does not cover show no
// alert at all.
//
// Because alerts come from the file, they exist for every position -- no
// tree coverage requirement -- and editing the file refreshes them without a
// restart (systemReady re-reads on mtime change). When the file is missing
// or unparsable, boards are simply served without alerts and everything
// else is unaffected.

// bidAlert is the client-visible alert for one call in one position: the
// system file's meaning of the call.
type bidAlert struct {
	Text       string `json:"text"`
	Artificial bool   `json:"artificial,omitempty"` // convention call: show the ⚠ marker
}

// callAlerts returns one alert per made call (parallel to calls) describing
// what each bot bid shows per the system file. Entries are nil for human
// seats (bit s set in humanMask) and for calls the file attaches no meaning
// to; an all-nil result collapses to nil so the field is omitted from the
// payload. The auction walks forward with the pattern-layer state and the
// structural reader, so pattern entries see the same context the trainer's
// matcher sees.
func (s *systemSpec) callAlerts(dealer int, calls []string, humanMask uint8) []any {
	out := make([]any, len(calls))
	found := false
	st := newMatchState(dealer)
	for i, c := range calls {
		id, ok := vocabID(c)
		if !ok {
			break
		}
		if seat := (dealer + i) % 4; humanMask&(1<<uint(seat)) == 0 {
			if a := s.alertForCall(st, int(id)); a != nil {
				out[i] = a
				found = true
			}
		}
		foldState(s.entries, dealer, st, i, int(id))
	}
	if !found {
		return nil
	}
	return out
}

// optionAlerts returns one alert per legal call (parallel to legal) for the
// position after calls (dealer-relative): what bidding it would show per the
// system file.
func (s *systemSpec) optionAlerts(dealer int, calls, legal []string) []any {
	st := newMatchState(dealer)
	for i, c := range calls {
		id, ok := vocabID(c)
		if !ok {
			return nil
		}
		foldState(s.entries, dealer, st, i, int(id))
	}
	out := make([]any, len(legal))
	found := false
	for i, c := range legal {
		id, ok := vocabID(c)
		if !ok {
			continue
		}
		if a := s.alertForCall(st, int(id)); a != nil {
			out[i] = a
			found = true
		}
	}
	if !found {
		return nil
	}
	return out
}
