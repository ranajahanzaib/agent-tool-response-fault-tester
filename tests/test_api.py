import httpx
import pytest

from agent_tool_response_fault_tester.api import app


@pytest.mark.asyncio
async def test_health_and_real_scenario_listing():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/api/health")).json() == {"status": "ok"}
        response = await client.get("/api/scenarios")
        assert response.status_code == 200
        assert response.json()[0]["id"] == "order-fulfillment"


@pytest.mark.asyncio
async def test_missing_scenario_is_404():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/scenarios/does-not-exist")
        assert response.status_code == 404
