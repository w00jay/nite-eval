// Hidden tests for coding_wine_medium_01, installed after the conversation.
// Dependencies are injected, so nothing here touches the network.
import { createHandler, SIMILARITY_THRESHOLD, type Deps, type Wine, type WineData } from "./scan_label.ts";

// Assertions are defined locally rather than imported from jsr:@std/assert:
// the sandbox has no network, so a remote import would fail to resolve and
// every test would error before running.
function assertEquals(actual: unknown, expected: unknown, msg?: string): void {
  const a = JSON.stringify(actual);
  const b = JSON.stringify(expected);
  if (a !== b) {
    throw new Error(`${msg ?? "assertEquals failed"}: got ${a}, want ${b}`);
  }
}

function assert(cond: boolean, msg = "assertion failed"): void {
  if (!cond) throw new Error(msg);
}

const EXTRACTED: WineData = {
  producer: "Cloudy Bay",
  wine_name: "Sauvignon Blanc",
  vintage: 2022,
  region: "Marlborough",
  varietal: "Sauvignon Blanc",
  wine_type: "white",
};

const MATCH: Wine = { id: "wine-1", ...EXTRACTED };

function deps(over: Partial<Deps> = {}): Deps {
  return {
    uploadImage: () => Promise.resolve({ url: "https://storage/img.jpg" }),
    extractLabel: () => Promise.resolve(EXTRACTED),
    findSimilarWine: () => Promise.resolve(null),
    createWine: (data) => Promise.resolve({ id: "new-1", ...data }),
    ...over,
  };
}

const IMAGE = btoa("fake-image-bytes");

function post(body: unknown): Request {
  return new Request("http://localhost/scan-label", {
    method: "POST",
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

async function json(res: Response): Promise<Record<string, unknown>> {
  return await res.json() as Record<string, unknown>;
}

Deno.test("matches an existing wine above the threshold", async () => {
  const handler = createHandler(deps({
    findSimilarWine: () => Promise.resolve({ wine: MATCH, score: 0.92 }),
  }));
  const res = await handler(post({ image: IMAGE }));
  assertEquals(res.status, 200);
  const body = await json(res);
  assertEquals(body.matched, true);
  assertEquals((body.wine as Wine).id, "wine-1");
});

Deno.test("creates a new wine when nothing matches", async () => {
  const captured: { created?: boolean } = {};
  const handler = createHandler(deps({
    findSimilarWine: () => Promise.resolve(null),
    createWine: (data) => {
      captured.created = true;
      return Promise.resolve({ id: "new-1", ...data });
    },
  }));
  const res = await handler(post({ image: IMAGE }));
  assertEquals(res.status, 200);
  const body = await json(res);
  assertEquals(body.matched, false);
  assert(captured.created === true, "createWine was not called");
});

Deno.test("a score below the threshold creates rather than matches", async () => {
  const handler = createHandler(deps({
    findSimilarWine: () => Promise.resolve({ wine: MATCH, score: SIMILARITY_THRESHOLD - 0.01 }),
  }));
  const body = await json(await handler(post({ image: IMAGE })));
  assertEquals(body.matched, false);
});

Deno.test("a score exactly at the threshold matches", async () => {
  const handler = createHandler(deps({
    findSimilarWine: () => Promise.resolve({ wine: MATCH, score: SIMILARITY_THRESHOLD }),
  }));
  const body = await json(await handler(post({ image: IMAGE })));
  assertEquals(body.matched, true);
});

Deno.test("a failing Claude call returns 200 with manual_entry_required", async () => {
  const handler = createHandler(deps({
    extractLabel: () => Promise.reject(new Error("claude exploded")),
  }));
  const res = await handler(post({ image: IMAGE }));
  assertEquals(res.status, 200, "a Claude failure must not surface as an error status");
  const body = await json(res);
  assertEquals(body.manual_entry_required, true);
});

Deno.test("a missing image is a 400 with a message", async () => {
  const handler = createHandler(deps());
  const res = await handler(post({}));
  assertEquals(res.status, 400);
  const body = await json(res);
  assert(typeof body.error === "string" && body.error.length > 0, "expected an error message");
});

Deno.test("an empty image is a 400", async () => {
  const res = await createHandler(deps())(post({ image: "" }));
  assertEquals(res.status, 400);
});

Deno.test("a non-string image is a 400", async () => {
  const res = await createHandler(deps())(post({ image: 12345 }));
  assertEquals(res.status, 400);
});

Deno.test("malformed JSON is a 400, not a crash", async () => {
  const res = await createHandler(deps())(post("{not json"));
  assertEquals(res.status, 400);
});

Deno.test("partial extraction is accepted", async () => {
  const handler = createHandler(deps({
    extractLabel: () => Promise.resolve({ ...EXTRACTED, vintage: null, region: null }),
  }));
  const res = await handler(post({ image: IMAGE }));
  assertEquals(res.status, 200);
  const body = await json(res);
  assertEquals((body.wine as Wine).vintage, null);
});

Deno.test("a storage failure is not fatal", async () => {
  // Captured in a holder: assigning to a `let` inside a callback lets
  // TypeScript narrow the outer binding to a type the assertions reject.
  const captured: { url?: string | null } = {};
  const handler = createHandler(deps({
    uploadImage: () => Promise.reject(new Error("bucket offline")),
    createWine: (data, url) => {
      captured.url = url;
      return Promise.resolve({ id: "new-1", ...data });
    },
  }));
  const res = await handler(post({ image: IMAGE }));
  assertEquals(res.status, 200, "storage is not fatal");
  assertEquals(captured.url, null, "createWine should receive null when upload failed");
});

Deno.test("the extracted metadata reaches createWine", async () => {
  const captured: { data?: WineData } = {};
  const handler = createHandler(deps({
    createWine: (data) => {
      captured.data = data;
      return Promise.resolve({ id: "new-1", ...data });
    },
  }));
  await handler(post({ image: IMAGE }));
  assertEquals(captured.data?.producer, "Cloudy Bay");
  assertEquals(captured.data?.vintage, 2022);
});
