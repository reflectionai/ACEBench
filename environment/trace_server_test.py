#!/usr/bin/env python3
"""Test script for Reset and Step RPCs."""

import asyncio
import grpc
import pytest
from proto.generated import env_rpc_pb2, env_rpc_pb2_grpc


@pytest.mark.asyncio
async def test_reset_and_step():
    """Test both Reset and Step RPCs in sequence."""
    async with grpc.aio.insecure_channel("127.0.0.1:50051") as channel:
        stub = env_rpc_pb2_grpc.EnvRpcStub(channel)

        # Test Reset first
        print("=== Testing Reset ===")
        ace_info = env_rpc_pb2.AceBenchResetRequest(
            model_name="gpt-4o",
            test_category="data_normal_single_turn_single_function",
            test_number=0,
            temperature=0.0,
        )

        reset_request = env_rpc_pb2.ResetRequest()
        reset_request.Extensions[env_rpc_pb2.ace_bench_reset_request_info].CopyFrom(
            ace_info
        )

        reset_response = await stub.Reset(reset_request)
        assert reset_response is not None

        # Test Step after successful reset
        print("=== Testing Step ===")
        step_request = env_rpc_pb2.StepRequest()
        step_response = await stub.Step(step_request)
        assert step_response is not None


# Keep the direct run capability too
async def main():
    await test_reset_and_step()


if __name__ == "__main__":
    asyncio.run(main())
