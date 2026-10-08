package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAssetVersionStable(t *testing.T) {
	a := assetVersion()
	b := assetVersion()
	if a == "" || a != b {
		t.Fatalf("asset version unstable: %q vs %q", a, b)
	}
	if len(a) != 12 {
		t.Fatalf("unexpected version length: %q", a)
	}
}

func TestServeHTMLStampsVersionAndCaching(t *testing.T) {
	h := serveHTML("static/play.html")
	rr := httptest.NewRecorder()
	h(rr, httptest.NewRequest("GET", "/play", nil))
	if rr.Code != 200 {
		t.Fatalf("status %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "text/html") {
		t.Fatalf("content type %q", ct)
	}
	if cc := rr.Header().Get("Cache-Control"); cc != "no-cache" {
		t.Fatalf("cache-control %q", cc)
	}
	body := rr.Body.String()
	ver := assetVersion()
	if !strings.Contains(body, "/static/shared.css?v="+ver) {
		t.Fatalf("css reference not stamped with %s", ver)
	}
	if !strings.Contains(body, "/static/shared.js?v="+ver) {
		t.Fatal("js reference not stamped")
	}
	if strings.Contains(body, `"/static/shared.css"`) {
		t.Fatal("unversioned reference left behind")
	}

	// a matching If-None-Match revalidates with an empty 304
	rr2 := httptest.NewRecorder()
	req2 := httptest.NewRequest("GET", "/play", nil)
	req2.Header.Set("If-None-Match", rr.Header().Get("ETag"))
	h(rr2, req2)
	if rr2.Code != http.StatusNotModified {
		t.Fatalf("want 304, got %d", rr2.Code)
	}
	if rr2.Body.Len() != 0 {
		t.Fatal("304 must not carry a body")
	}
}

func TestServeSWJSContentType(t *testing.T) {
	h := serveHTML("static/sw.js")
	rr := httptest.NewRecorder()
	h(rr, httptest.NewRequest("GET", "/sw.js", nil))
	if rr.Code != 200 {
		t.Fatalf("status %d", rr.Code)
	}
	// nosniff is set globally, so a wrong type would kill the service worker
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "javascript") {
		t.Fatalf("sw.js content type %q", ct)
	}
	if cc := rr.Header().Get("Cache-Control"); cc != "no-cache" {
		t.Fatalf("sw.js cache-control %q", cc)
	}
}

func TestStaticFileHandlerCaching(t *testing.T) {
	h := staticFileHandler()
	ver := assetVersion()

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest("GET", "/static/shared.css?v="+ver, nil))
	if rr.Code != 200 {
		t.Fatalf("status %d", rr.Code)
	}
	if cc := rr.Header().Get("Cache-Control"); !strings.Contains(cc, "immutable") {
		t.Fatalf("versioned asset cache-control %q", cc)
	}

	for _, q := range []string{"", "?v=staleversion"} {
		rr2 := httptest.NewRecorder()
		h.ServeHTTP(rr2, httptest.NewRequest("GET", "/static/shared.css"+q, nil))
		if rr2.Code != 200 {
			t.Fatalf("status %d for %q", rr2.Code, q)
		}
		if cc := rr2.Header().Get("Cache-Control"); cc != "no-cache" {
			t.Fatalf("unversioned (%q) cache-control %q", q, cc)
		}
	}
}
