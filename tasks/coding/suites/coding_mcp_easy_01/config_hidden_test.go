// Hidden test suite for coding_mcp_easy_01. Installed after the conversation
// ends, so the model never sees these cases and cannot tune to them.
//
// Compiles against the contract stated in the task prompt:
//
//	package config
//	func Load(path string) (*Config, error)
//	func (c *Config) EnabledServers() []ServerConfig
//	type ServerConfig struct { Name, URL, Transport, AuthType string; Enabled bool }
package config

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "servers.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing fixture: %v", err)
	}
	return path
}

func TestHiddenLoadValidConfig(t *testing.T) {
	path := writeTemp(t, `servers:
  - name: notion
    url: https://mcp.notion.com/mcp
    transport: sse
    auth_type: bearer
    auth_token: "ntn_test123"
    enabled: true
  - name: wooj-brain
    url: https://example.supabase.co/functions/v1/mcp
    transport: http
    auth_type: api_key
    api_key: "sk_test456"
    enabled: true
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	if len(cfg.Servers) != 2 {
		t.Fatalf("expected 2 servers, got %d", len(cfg.Servers))
	}
	if cfg.Servers[0].Name != "notion" || cfg.Servers[0].AuthType != "bearer" {
		t.Errorf("server 0 wrong: %+v", cfg.Servers[0])
	}
	if cfg.Servers[1].Transport != "http" || cfg.Servers[1].AuthType != "api_key" {
		t.Errorf("server 1 wrong: %+v", cfg.Servers[1])
	}
}

func TestHiddenInvalidURLRejected(t *testing.T) {
	path := writeTemp(t, `servers:
  - name: bad
    url: "not-a-url"
    transport: sse
    auth_type: bearer
`)
	if _, err := Load(path); err == nil {
		t.Fatal("expected a validation error for a malformed URL, got nil")
	}
}

func TestHiddenUnknownAuthTypeRejected(t *testing.T) {
	path := writeTemp(t, `servers:
  - name: bad
    url: https://example.com
    transport: sse
    auth_type: "magic"
`)
	if _, err := Load(path); err == nil {
		t.Fatal("expected a validation error for an unknown auth type, got nil")
	}
}

func TestHiddenEnabledServersFilters(t *testing.T) {
	path := writeTemp(t, `servers:
  - name: active
    url: https://a.com
    transport: sse
    auth_type: bearer
    enabled: true
  - name: disabled
    url: https://b.com
    transport: sse
    auth_type: bearer
    enabled: false
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	enabled := cfg.EnabledServers()
	if len(enabled) != 1 || enabled[0].Name != "active" {
		t.Fatalf("expected only the active server, got %+v", enabled)
	}
}

// Cases the visible task description does not enumerate. A model that hardcodes
// for the four listed cases fails here.

func TestHiddenMissingFileIsAnError(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "does-not-exist.yaml")); err == nil {
		t.Fatal("expected an error for a missing config file, got nil")
	}
}

func TestHiddenMalformedYAMLIsAnError(t *testing.T) {
	path := writeTemp(t, "servers:\n  - name: [unclosed\n")
	if _, err := Load(path); err == nil {
		t.Fatal("expected an error for malformed YAML, got nil")
	}
}

func TestHiddenUnknownTransportRejected(t *testing.T) {
	path := writeTemp(t, `servers:
  - name: bad
    url: https://example.com
    transport: carrier-pigeon
    auth_type: bearer
`)
	if _, err := Load(path); err == nil {
		t.Fatal("expected a validation error for an unknown transport, got nil")
	}
}

func TestHiddenEmptyServerListLoadsCleanly(t *testing.T) {
	path := writeTemp(t, "servers: []\n")
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("an empty server list is valid, got error: %v", err)
	}
	if len(cfg.EnabledServers()) != 0 {
		t.Errorf("expected no enabled servers, got %d", len(cfg.EnabledServers()))
	}
}
