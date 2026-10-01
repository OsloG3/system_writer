package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/klauspost/compress/zstd"
)

// ---- Bid alerts from the model's self-play tree ----
//
// models/last.json.zst is the auction prefix tree dumped by bidding-dt's
// analyze tool from the same checkpoint (models/last.pt) the play sidecar
// drives the bots with. For every call in every sequence from the dealer it
// records how many self-play deals the model made that call in that position
// (n) plus the min/max HCP and per-suit lengths of the hand it held --
// exactly an alert text for the bid, and n over the parent's n is how often
// the model picks that call there.
//
// The raw JSON is hundreds of MB, so it ships zstd-compressed (~1.9 MB,
// level 19) and is never unmarshalled into maps: a token streaming decoder
// compiles it once into two flat arenas (~5 MB for the 163k-node tree), the
// arenas are cached as a binary file next to the source and reloaded while
// the source's size/mtime are unchanged. Plain .json and .gz sources are
// sniffed and accepted too. Loading runs on a background goroutine; until it
// finishes (or when the file is missing) boards are simply served without
// alerts and everything else is unaffected.

const (
	defaultAlertTreeFile = "models/last.json" // ALERT_TREE_JSON overrides
	alertBinVersion      = uint32(1)
	alertBinHeaderSize   = 48
	alertNodeSize        = 20 // packed bytes per node in the cache
	alertKidSize         = 8  // packed bytes per child edge in the cache
	alertMaxNodes        = 16 << 20
	alertMaxDepth        = 64 // auctions are capped at MAX_CALLS = 36
	alertSuitCount       = 4  // S,H,D,C as in bidding_dt.data.hands
)

var alertBinMagic = []byte("ALRTREE\x00")

// alertTreePtr holds the compiled tree once ready; nil until then.
var alertTreePtr atomic.Pointer[alertTree]

// vocabID maps a canonical call (P/X/XX/1C..7N) onto its bidding-dt token id
// (vocab.py: P=1, X=2, XX=3, bids ordered by level with denoms C<D<H<S<N).
func vocabID(call string) (uint8, bool) {
	switch call {
	case "P":
		return 1, true
	case "X":
		return 2, true
	case "XX":
		return 3, true
	}
	if len(call) == 2 && call[0] >= '1' && call[0] <= '7' {
		if d := strings.IndexByte("CDHSN", call[1]); d >= 0 {
			return 4 + (call[0]-'1')*5 + uint8(d), true
		}
	}
	return 0, false
}

// alertNode is one auction position's summary: the bidder's pooled hand
// ranges and how often the model reached it. Children live contiguously in
// alertTree.kids, sorted by token id.
type alertNode struct {
	n        int32
	childOff int32
	kids     uint8
	seen     uint8 // hand stats observed (hcp/suits non-null in the JSON)
	hcpLo    uint8
	hcpHi    uint8
	suitLo   [alertSuitCount]uint8 // S,H,D,C
	suitHi   [alertSuitCount]uint8
}

type alertKid struct {
	call uint8
	_    [3]byte
	node int32
}

type alertTree struct {
	nodes []alertNode
	kids  []alertKid
	deals int32 // self-play deals behind the root visit count
}

// bidAlert is the client-visible alert for one call in one position.
type bidAlert struct {
	N     int32      `json:"n"`
	Pct   float64    `json:"pct"`   // share of the parent position's visits
	HCP   [2]int16   `json:"hcp"`   // pooled min/max of the bidder's hand
	Suits [4][2]int8 `json:"suits"` // min/max length per suit, S,H,D,C
}

// alertTreePath resolves the tree source: the explicit ALERT_TREE_JSON
// override, else the first of .zst / .gz / plain .json that exists.
func alertTreePath() string {
	if p := strings.TrimSpace(os.Getenv("ALERT_TREE_JSON")); p != "" {
		return p
	}
	for _, cand := range []string{defaultAlertTreeFile + ".zst", defaultAlertTreeFile + ".gz", defaultAlertTreeFile} {
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	return defaultAlertTreeFile
}

// alertBinPath is the compiled-cache path for a source file: last.json.zst,
// last.json.gz and last.json all share models/last.bin.
func alertBinPath(path string) string {
	return strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(path, ".zst"), ".gz"), ".json") + ".bin"
}

func initAlerts() {
	path := alertTreePath()
	go func() {
		start := time.Now()
		t, err := loadAlertTree(path)
		if err != nil {
			log.Printf("Bid alerts unavailable (%s): %v", path, err)
			return
		}
		alertTreePtr.Store(t)
		log.Printf("Bid alerts ready: %d nodes from %d self-play deals, loaded in %s",
			len(t.nodes), t.deals, time.Since(start).Round(time.Millisecond))
	}()
}

// alertTreeReady returns the compiled tree, or nil while it is still loading
// or when it could not be built.
func alertTreeReady() *alertTree { return alertTreePtr.Load() }

func loadAlertTree(path string) (*alertTree, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	binPath := alertBinPath(path)
	if t, err := readAlertBin(binPath, fi); err == nil {
		return t, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		log.Printf("Alert tree cache unusable (%s), recompiling from JSON", err)
	}
	t, err := compileAlertTree(path)
	if err != nil {
		return nil, err
	}
	if err := writeAlertBin(binPath, t, fi); err != nil {
		log.Printf("Could not cache the alert tree in %s: %v", binPath, err)
	}
	return t, nil
}

// ---- Lookups ----

// child returns the node reached from parent by the call, or -1.
func (t *alertTree) child(parent int32, call string) int32 {
	id, ok := vocabID(call)
	if !ok || parent < 0 || parent >= int32(len(t.nodes)) {
		return -1
	}
	nd := &t.nodes[parent]
	kids := t.kids[nd.childOff : int32(nd.childOff)+int32(nd.kids)]
	i := sort.Search(len(kids), func(j int) bool { return kids[j].call >= id })
	if i < len(kids) && kids[i].call == id {
		return kids[i].node
	}
	return -1
}

// walk returns the node after the whole call sequence, or -1.
func (t *alertTree) walk(calls []string) int32 {
	idx := int32(0)
	for _, c := range calls {
		idx = t.child(idx, c)
		if idx < 0 {
			return -1
		}
	}
	return idx
}

// alertAt renders the alert for node, reached from parent (pct denominator).
func (t *alertTree) alertAt(parent, node int32) *bidAlert {
	if node < 0 || node >= int32(len(t.nodes)) {
		return nil
	}
	nd := &t.nodes[node]
	if nd.seen == 0 {
		return nil
	}
	a := &bidAlert{N: nd.n, HCP: [2]int16{int16(nd.hcpLo), int16(nd.hcpHi)}}
	for i := 0; i < alertSuitCount; i++ {
		a.Suits[i] = [2]int8{int8(nd.suitLo[i]), int8(nd.suitHi[i])}
	}
	if parent >= 0 && parent < int32(len(t.nodes)) && t.nodes[parent].n > 0 {
		a.Pct = math.Round(float64(nd.n)/float64(t.nodes[parent].n)*1e4) / 1e4
	}
	return a
}

// callAlerts returns one alert per made call (parallel to calls) describing
// what each bot bid promised. Entries are nil for human seats (bit s set in
// humanMask) and for calls the self-play tree holds no data for; an all-nil
// result collapses to nil so the field is omitted from the payload.
func (t *alertTree) callAlerts(dealer int, calls []string, humanMask uint8) []any {
	out := make([]any, len(calls))
	found := false
	idx := int32(0)
	for i, c := range calls {
		next := t.child(idx, c)
		if next < 0 {
			break
		}
		if seat := (dealer + i) % 4; humanMask&(1<<uint(seat)) == 0 {
			if a := t.alertAt(idx, next); a != nil {
				out[i] = a
				found = true
			}
		}
		idx = next
	}
	if !found {
		return nil
	}
	return out
}

// optionAlerts returns one alert per legal call (parallel to legal) for the
// position after calls: what bidding it would show and how often the model
// picks it there. Nil when the tree has no data for the position.
func (t *alertTree) optionAlerts(calls, legal []string) []any {
	pos := t.walk(calls)
	if pos < 0 {
		return nil
	}
	out := make([]any, len(legal))
	found := false
	for i, c := range legal {
		if a := t.alertAt(pos, t.child(pos, c)); a != nil {
			out[i] = a
			found = true
		}
	}
	if !found {
		return nil
	}
	return out
}

// ---- Streaming JSON compiler ----

type treeParser struct {
	dec   *json.Decoder
	nodes []alertNode
	kids  []alertKid
}

// openTreeSource wraps the open file in the decompressor its magic bytes
// call for: zstd (28 B5 2F FD), gzip (1F 8B) or plain JSON. The returned
// function releases the decompressor.
func openTreeSource(f *os.File) (io.Reader, func(), error) {
	br := bufio.NewReaderSize(f, 1<<20)
	head, err := br.Peek(2)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, nil, err
	}
	if len(head) == 2 && head[0] == 0x1f && head[1] == 0x8b {
		zr, err := gzip.NewReader(br)
		if err != nil {
			return nil, nil, err
		}
		return zr, func() { zr.Close() }, nil
	}
	head4, err := br.Peek(4)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, nil, err
	}
	if len(head4) == 4 && head4[0] == 0x28 && head4[1] == 0xb5 && head4[2] == 0x2f && head4[3] == 0xfd {
		zr, err := zstd.NewReader(br)
		if err != nil {
			return nil, nil, err
		}
		return zr, zr.Close, nil
	}
	return br, func() {}, nil
}

// compileAlertTree token-streams the (optionally compressed) report file
// into the flat arenas. Peak memory stays within a few tens of MB regardless
// of the JSON's size.
func compileAlertTree(path string) (*alertTree, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	src, release, err := openTreeSource(f)
	if err != nil {
		return nil, err
	}
	defer release()
	p := &treeParser{dec: json.NewDecoder(src)}
	p.dec.UseNumber()
	return p.parseReport()
}

func tokenInt(tok any) (int64, bool) {
	switch v := tok.(type) {
	case json.Number:
		i, err := v.Int64()
		return i, err == nil
	case float64:
		return int64(v), true
	}
	return 0, false
}

func (p *treeParser) parseReport() (*alertTree, error) {
	tok, err := p.dec.Token()
	if err != nil {
		return nil, err
	}
	if tok != json.Delim('{') {
		return nil, errors.New("expected a JSON object at the top level")
	}
	out := &alertTree{}
	treeIdx := int32(-1)
	for {
		tok, err = p.dec.Token()
		if err != nil {
			return nil, err
		}
		if tok == json.Delim('}') {
			break
		}
		key, ok := tok.(string)
		if !ok {
			return nil, errors.New("expected a key in the report object")
		}
		switch key {
		case "deals":
			tok, err = p.dec.Token()
			if err != nil {
				return nil, err
			}
			if n, ok := tokenInt(tok); ok {
				out.deals = int32(n)
			}
		case "tree":
			if treeIdx >= 0 {
				return nil, errors.New("duplicate tree key")
			}
			if treeIdx, err = p.parseNode(0); err != nil {
				return nil, err
			}
		default:
			if err = p.skipValue(); err != nil {
				return nil, err
			}
		}
	}
	if treeIdx != 0 || len(p.nodes) == 0 {
		return nil, errors.New("no tree found in the report")
	}
	out.nodes, out.kids = p.nodes, p.kids
	if out.deals == 0 {
		out.deals = out.nodes[0].n
	}
	return out, nil
}

// parseNode consumes one node object (the caller sits before its '{') and
// returns its arena index. Child subtrees are parsed first, then this node's
// kid block is appended, keeping every node's children contiguous.
func (p *treeParser) parseNode(depth int) (int32, error) {
	if depth > alertMaxDepth {
		return -1, errors.New("tree nests deeper than any legal auction")
	}
	tok, err := p.dec.Token()
	if err != nil {
		return -1, err
	}
	if tok != json.Delim('{') {
		return -1, errors.New("expected a node object")
	}
	if len(p.nodes) >= alertMaxNodes {
		return -1, errors.New("tree exceeds the supported node count")
	}
	idx := int32(len(p.nodes))
	p.nodes = append(p.nodes, alertNode{})
	var nd alertNode
	var kids []alertKid
	for {
		tok, err = p.dec.Token()
		if err != nil {
			return -1, err
		}
		if tok == json.Delim('}') {
			break
		}
		key, ok := tok.(string)
		if !ok {
			return -1, errors.New("expected a key in the node object")
		}
		switch key {
		case "call":
			if _, err = p.dec.Token(); err != nil { // name or null; implied by the position
				return -1, err
			}
		case "n":
			if tok, err = p.dec.Token(); err != nil {
				return -1, err
			}
			if n, ok := tokenInt(tok); ok {
				nd.n = int32(n)
			}
		case "hcp":
			lo, hi, seen, err := p.parseRange()
			if err != nil {
				return -1, err
			}
			if seen {
				nd.seen = 1
				nd.hcpLo, nd.hcpHi = uint8(lo), uint8(hi)
			}
		case "suits":
			if err := p.parseSuits(&nd); err != nil {
				return -1, err
			}
		case "children":
			if kids, err = p.parseChildren(depth); err != nil {
				return -1, err
			}
		default:
			if err := p.skipValue(); err != nil {
				return -1, err
			}
		}
	}
	sort.Slice(kids, func(i, j int) bool { return kids[i].call < kids[j].call })
	nd.childOff = int32(len(p.kids))
	nd.kids = uint8(min(len(kids), 255))
	p.kids = append(p.kids, kids...)
	p.nodes[idx] = nd
	return idx, nil
}

// parseRange consumes null or [lo, hi].
func (p *treeParser) parseRange() (lo, hi int64, seen bool, err error) {
	tok, err := p.dec.Token()
	if err != nil {
		return 0, 0, false, err
	}
	if tok == nil {
		return 0, 0, false, nil
	}
	if tok != json.Delim('[') {
		return 0, 0, false, errors.New("expected a [lo, hi] range")
	}
	var a, b any
	if a, err = p.dec.Token(); err != nil {
		return 0, 0, false, err
	}
	if b, err = p.dec.Token(); err != nil {
		return 0, 0, false, err
	}
	if tok, err = p.dec.Token(); err != nil {
		return 0, 0, false, err
	}
	if tok != json.Delim(']') {
		return 0, 0, false, errors.New("unterminated range")
	}
	lo, ok1 := tokenInt(a)
	hi, ok2 := tokenInt(b)
	return lo, hi, ok1 && ok2, nil
}

// parseSuits consumes null or {"S": [lo,hi], "H": ..., "D": ..., "C": ...}.
func (p *treeParser) parseSuits(nd *alertNode) error {
	tok, err := p.dec.Token()
	if err != nil {
		return err
	}
	if tok == nil {
		return nil
	}
	if tok != json.Delim('{') {
		return errors.New("expected a suits object")
	}
	suitIdx := map[string]int{"S": 0, "H": 1, "D": 2, "C": 3}
	for {
		tok, err = p.dec.Token()
		if err != nil {
			return err
		}
		if tok == json.Delim('}') {
			return nil
		}
		key, _ := tok.(string)
		lo, hi, seen, err := p.parseRange()
		if err != nil {
			return err
		}
		if i, ok := suitIdx[key]; ok && seen {
			nd.seen = 1
			nd.suitLo[i], nd.suitHi[i] = uint8(lo), uint8(hi)
		}
	}
}

// parseChildren consumes the children object, recursing into every child.
func (p *treeParser) parseChildren(depth int) ([]alertKid, error) {
	tok, err := p.dec.Token()
	if err != nil {
		return nil, err
	}
	if tok == nil {
		return nil, nil
	}
	if tok != json.Delim('{') {
		return nil, errors.New("expected a children object")
	}
	var kids []alertKid
	for {
		tok, err = p.dec.Token()
		if err != nil {
			return nil, err
		}
		if tok == json.Delim('}') {
			return kids, nil
		}
		name, ok := tok.(string)
		if !ok {
			return nil, errors.New("expected a call name in children")
		}
		child, err := p.parseNode(depth + 1)
		if err != nil {
			return nil, err
		}
		if id, ok := vocabID(name); ok {
			kids = append(kids, alertKid{call: id, node: child})
		}
	}
}

// skipValue consumes one JSON value of any shape.
func (p *treeParser) skipValue() error {
	tok, err := p.dec.Token()
	if err != nil {
		return err
	}
	d, ok := tok.(json.Delim)
	if !ok {
		return nil // scalar, already consumed
	}
	if d == '}' || d == ']' {
		return fmt.Errorf("unexpected %s while skipping a value", d)
	}
	for depth := 1; depth > 0; {
		tok, err = p.dec.Token()
		if err != nil {
			return err
		}
		if d, ok := tok.(json.Delim); ok {
			switch d {
			case '{', '[':
				depth++
			case '}', ']':
				depth--
			}
		}
	}
	return nil
}

// ---- Binary cache (compiled arenas, keyed by the source file's stat) ----
//
// Header (48 bytes, little endian): magic[8], version u32, reserved u32,
// source size u64, source mtime ns u64, node count u32, kid count u32,
// deals u32, reserved u32. Then the packed node and kid arenas.

func writeAlertBin(path string, t *alertTree, fi os.FileInfo) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	fail := func(err error) error {
		f.Close()
		os.Remove(tmp)
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	hdr := make([]byte, alertBinHeaderSize)
	copy(hdr, alertBinMagic)
	binary.LittleEndian.PutUint32(hdr[8:], alertBinVersion)
	binary.LittleEndian.PutUint64(hdr[16:], uint64(fi.Size()))
	binary.LittleEndian.PutUint64(hdr[24:], uint64(fi.ModTime().UnixNano()))
	binary.LittleEndian.PutUint32(hdr[32:], uint32(len(t.nodes)))
	binary.LittleEndian.PutUint32(hdr[36:], uint32(len(t.kids)))
	binary.LittleEndian.PutUint32(hdr[40:], uint32(t.deals))
	if _, err := w.Write(hdr); err != nil {
		return fail(err)
	}
	nodes := make([]byte, len(t.nodes)*alertNodeSize)
	for i, n := range t.nodes {
		o := i * alertNodeSize
		binary.LittleEndian.PutUint32(nodes[o:], uint32(n.n))
		binary.LittleEndian.PutUint32(nodes[o+4:], uint32(n.childOff))
		nodes[o+8] = n.kids
		nodes[o+9] = n.seen
		nodes[o+10] = n.hcpLo
		nodes[o+11] = n.hcpHi
		copy(nodes[o+12:], n.suitLo[:])
		copy(nodes[o+16:], n.suitHi[:])
	}
	if _, err := w.Write(nodes); err != nil {
		return fail(err)
	}
	kids := make([]byte, len(t.kids)*alertKidSize)
	for i, k := range t.kids {
		o := i * alertKidSize
		kids[o] = k.call
		binary.LittleEndian.PutUint32(kids[o+4:], uint32(k.node))
	}
	if _, err := w.Write(kids); err != nil {
		return fail(err)
	}
	if err := w.Flush(); err != nil {
		return fail(err)
	}
	if err := f.Close(); err != nil {
		return fail(err)
	}
	return os.Rename(tmp, path)
}

func readAlertBin(path string, fi os.FileInfo) (*alertTree, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < alertBinHeaderSize || string(data[:8]) != string(alertBinMagic) {
		return nil, errors.New("not an alert tree cache")
	}
	if v := binary.LittleEndian.Uint32(data[8:12]); v != alertBinVersion {
		return nil, fmt.Errorf("cache version %d, want %d", v, alertBinVersion)
	}
	if size := int64(binary.LittleEndian.Uint64(data[16:24])); size != fi.Size() {
		return nil, errors.New("source file changed size")
	}
	if mod := int64(binary.LittleEndian.Uint64(data[24:32])); mod != fi.ModTime().UnixNano() {
		return nil, errors.New("source file changed")
	}
	nNodes := binary.LittleEndian.Uint32(data[32:36])
	nKids := binary.LittleEndian.Uint32(data[36:40])
	if nNodes == 0 || nNodes > alertMaxNodes || nKids > alertMaxNodes {
		return nil, errors.New("implausible cache counts")
	}
	if len(data) != alertBinHeaderSize+int(nNodes)*alertNodeSize+int(nKids)*alertKidSize {
		return nil, errors.New("truncated cache")
	}
	t := &alertTree{
		deals: int32(binary.LittleEndian.Uint32(data[40:44])),
		nodes: make([]alertNode, nNodes),
		kids:  make([]alertKid, nKids),
	}
	buf := data[alertBinHeaderSize:]
	for i := range t.nodes {
		o := i * alertNodeSize
		n := &t.nodes[i]
		n.n = int32(binary.LittleEndian.Uint32(buf[o:]))
		n.childOff = int32(binary.LittleEndian.Uint32(buf[o+4:]))
		n.kids = buf[o+8]
		n.seen = buf[o+9]
		n.hcpLo = buf[o+10]
		n.hcpHi = buf[o+11]
		copy(n.suitLo[:], buf[o+12:o+16])
		copy(n.suitHi[:], buf[o+16:o+20])
	}
	buf = data[alertBinHeaderSize+int(nNodes)*alertNodeSize:]
	for i := range t.kids {
		o := i * alertKidSize
		t.kids[i] = alertKid{call: buf[o], node: int32(binary.LittleEndian.Uint32(buf[o+4:]))}
	}
	return t, nil
}
