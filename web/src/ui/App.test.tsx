import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Agent Tool Response Fault Tester dashboard", () => {
  it("loads scenarios from the API and enables an experiment", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        const payload = url.endsWith("/api/scenarios")
          ? [
              {
                id: "order-fulfillment",
                name: "Order Fulfillment",
                tools: [],
                faults: [
                  {
                    id: "fault-1",
                    name: "Missing field",
                    target: { tool: "verify" },
                    mutation: { type: "remove_field" },
                  },
                ],
              },
            ]
          : [];
        return new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
    render(<App />);
    expect(await screen.findByText("Order Fulfillment")).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: /run replay/i }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it("shows a useful API error instead of inventing results", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline");
      }),
    );
    render(<App />);
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(screen.getByText(/start the local api/i)).toBeTruthy();
  });
});
