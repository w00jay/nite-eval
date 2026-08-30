// Hidden test suite for coding_mcp_hard_01. Installed after the conversation
// ends, so the model never sees these cases.
//
// Upstreams are httptest servers, which bind loopback and work fine with the
// sandbox's --network none.
package gateway

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func upstream(t *testing.T, tools []string, record func(*http.Request)) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/tools/list", func(w http.ResponseWriter, r *http.Request) {
		if record != nil {
			record(r)
		}
		out := make([]Tool, 0, len(tools))
		for _, name := range tools {
			out = append(out, Tool{Name: name, Description: name + " description"})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"tools": out})
	})
	mux.HandleFunc("/tools/call", func(w http.ResponseWriter, r *http.Request) {
		if record != nil {
			record(r)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "served_by": r.Host})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func newGateway(t *testing.T, cfg *Config, buf *strings.Builder, timeout time.Duration) http.Handler {
	t.Helper()
	var logger *slog.Logger
	if buf != nil {
		logger = slog.New(slog.NewJSONHandler(buf, nil))
	} else {
		logger = slog.New(slog.NewJSONHandler(&strings.Builder{}, nil))
	}
	return New(cfg, &http.Client{Timeout: timeout}, logger).Handler()
}

func post(t *testing.T, h http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest(http.MethodPost, path, &buf)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestHiddenToolsListAggregatesAndDedupes(t *testing.T) {
	a := upstream(t, []string{"search", "shared"}, nil)
	b := upstream(t, []string{"shared", "create"}, nil)
	cfg := &Config{Servers: []ServerConfig{
		{Name: "a", BaseURL: a.URL, AuthType: "none"},
		{Name: "b", BaseURL: b.URL, AuthType: "none"},
	}}

	rec := post(t, newGateway(t, cfg, nil, 5*time.Second), "/mcp/tools/list", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var got struct{ Tools []Tool }
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding response: %v (body %s)", err, rec.Body.String())
	}
	names := map[string]int{}
	for _, tool := range got.Tools {
		names[tool.Name]++
	}
	if len(got.Tools) != 3 {
		t.Fatalf("expected 3 deduped tools, got %d: %+v", len(got.Tools), got.Tools)
	}
	if names["shared"] != 1 {
		t.Errorf("duplicate tool not deduplicated: %+v", got.Tools)
	}
}

func TestHiddenToolsCallRoutesToOwningUpstream(t *testing.T) {
	var aHits, bHits int
	a := upstream(t, []string{"only_a"}, func(r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/tools/call") {
			aHits++
		}
	})
	b := upstream(t, []string{"only_b"}, func(r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/tools/call") {
			bHits++
		}
	})
	cfg := &Config{Servers: []ServerConfig{
		{Name: "a", BaseURL: a.URL, AuthType: "none"},
		{Name: "b", BaseURL: b.URL, AuthType: "none"},
	}}
	h := newGateway(t, cfg, nil, 5*time.Second)

	if rec := post(t, h, "/mcp/tools/call", map[string]any{"tool": "only_b"}); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if bHits != 1 || aHits != 0 {
		t.Errorf("routed wrong: a=%d b=%d", aHits, bHits)
	}
}

func TestHiddenBearerAuthInjected(t *testing.T) {
	var seen string
	up := upstream(t, []string{"tool"}, func(r *http.Request) { seen = r.Header.Get("Authorization") })
	cfg := &Config{Servers: []ServerConfig{
		{Name: "a", BaseURL: up.URL, AuthType: "bearer", AuthToken: "tok_123"},
	}}
	post(t, newGateway(t, cfg, nil, 5*time.Second), "/mcp/tools/call", map[string]any{"tool": "tool"})
	if seen != "Bearer tok_123" {
		t.Errorf("Authorization header = %q, want %q", seen, "Bearer tok_123")
	}
}

func TestHiddenAPIKeyAuthInjected(t *testing.T) {
	var seen string
	up := upstream(t, []string{"tool"}, func(r *http.Request) { seen = r.Header.Get("X-API-Key") })
	cfg := &Config{Servers: []ServerConfig{
		{Name: "a", BaseURL: up.URL, AuthType: "api_key", APIKey: "key_456"},
	}}
	post(t, newGateway(t, cfg, nil, 5*time.Second), "/mcp/tools/call", map[string]any{"tool": "tool"})
	if seen != "key_456" {
		t.Errorf("X-API-Key header = %q, want %q", seen, "key_456")
	}
}

func TestHiddenUnknownToolReturns404(t *testing.T) {
	up := upstream(t, []string{"known"}, nil)
	cfg := &Config{Servers: []ServerConfig{{Name: "a", BaseURL: up.URL, AuthType: "none"}}}
	rec := post(t, newGateway(t, cfg, nil, 5*time.Second), "/mcp/tools/call", map[string]any{"tool": "nope"})
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404 (body %s)", rec.Code, rec.Body.String())
	}
}

func TestHiddenUpstreamTimeoutReturns504(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/tools/list") {
			_ = json.NewEncoder(w).Encode(map[string]any{"tools": []Tool{{Name: "slow"}}})
			return
		}
		time.Sleep(2 * time.Second)
	}))
	t.Cleanup(slow.Close)

	cfg := &Config{Servers: []ServerConfig{{Name: "slow", BaseURL: slow.URL, AuthType: "none"}}}
	rec := post(t, newGateway(t, cfg, nil, 200*time.Millisecond), "/mcp/tools/call", map[string]any{"tool": "slow"})
	if rec.Code != http.StatusGatewayTimeout {
		t.Errorf("status = %d, want 504 (body %s)", rec.Code, rec.Body.String())
	}
}

func TestHiddenConcurrentToolsListIsRaceFree(t *testing.T) {
	a := upstream(t, []string{"a1", "a2"}, nil)
	b := upstream(t, []string{"b1"}, nil)
	cfg := &Config{Servers: []ServerConfig{
		{Name: "a", BaseURL: a.URL, AuthType: "none"},
		{Name: "b", BaseURL: b.URL, AuthType: "none"},
	}}
	h := newGateway(t, cfg, nil, 5*time.Second)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rec := post(t, h, "/mcp/tools/list", nil)
			if rec.Code != http.StatusOK {
				t.Errorf("status = %d", rec.Code)
			}
		}()
	}
	wg.Wait()
}

func TestHiddenStructuredLogHasRequiredKeys(t *testing.T) {
	up := upstream(t, []string{"tool"}, nil)
	cfg := &Config{Servers: []ServerConfig{{Name: "srv", BaseURL: up.URL, AuthType: "none"}}}
	var logs strings.Builder
	post(t, newGateway(t, cfg, &logs, 5*time.Second), "/mcp/tools/call", map[string]any{"tool": "tool"})

	var found bool
	for _, line := range strings.Split(strings.TrimSpace(logs.String()), "\n") {
		if line == "" {
			continue
		}
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		_, hasServer := entry["server"]
		_, hasTool := entry["tool"]
		_, hasLatency := entry["latency_ms"]
		_, hasStatus := entry["status"]
		if hasServer && hasTool && hasLatency && hasStatus {
			found = true
		}
	}
	if !found {
		t.Errorf("no log line with server/tool/latency_ms/status; got:\n%s", logs.String())
	}
}
