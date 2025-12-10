"""Darkfield Cloud API client.

The Cloud API is the recommended way to use Darkfield for batch processing.
It handles:
- Activation collection at scale
- Access to pre-trained persona vectors
- Optimized GPU inference
- Detailed risk reports

Get an API key at https://darkfield.ai
"""

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class Job:
    """A Darkfield Cloud screening job."""

    id: str
    status: str
    report_url: str | None = None
    results: dict[str, Any] | None = None

    def is_complete(self) -> bool:
        """Check if job has finished."""
        return self.status in ("completed", "failed")

    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == "completed"


class Client:
    """Client for the Darkfield Cloud API.

    Example:
        client = Client(api_key="dk_live_...")

        # Submit a dataset for screening
        job = client.scan_dataset(
            file="training_data.jsonl",
            vectors=["sycophancy", "hallucination", "evil"],
        )

        # Check status
        job = client.get_job(job.id)
        print(job.status)  # "processing" -> "completed"

        # Get results
        if job.is_complete():
            print(job.report_url)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.darkfield.ai/v1",
    ):
        """Initialize the client.

        Args:
            api_key: Your Darkfield API key (get one at https://darkfield.ai)
            base_url: API base URL (default: production)
        """
        if not api_key:
            raise ValueError(
                "API key is required. Get one at https://darkfield.ai/signup"
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "darkfield-python-sdk/0.1.0",
        })

    def scan_dataset(
        self,
        file: str,
        vectors: list[str],
        mode: str = "raw",
        webhook_url: str | None = None,
    ) -> Job:
        """Submit a dataset for screening.

        Args:
            file: Path to the dataset file (JSONL format)
                  Each line should have "prompt" and "response" fields.
            vectors: List of trait names to screen against.
                     Options: sycophancy, hallucination, evil, refusal,
                     toxicity, deception, medical_misinfo, legal_misinfo,
                     financial_misinfo, pii_leakage, insecure_code, authority_overclaim
            mode: Screening mode - "raw" (fast) or "full" (with natural response comparison)
            webhook_url: Optional URL to POST results when complete

        Returns:
            Job object tracking the screening process

        Example:
            job = client.scan_dataset(
                file="my_training_data.jsonl",
                vectors=["sycophancy", "evil"],
            )
            print(f"Job submitted: {job.id}")
        """
        # Validate vectors
        valid_vectors = {
            "sycophancy", "hallucination", "evil", "refusal",
            "toxicity", "deception", "medical_misinfo", "legal_misinfo",
            "financial_misinfo", "pii_leakage", "insecure_code", "authority_overclaim",
        }
        invalid = set(vectors) - valid_vectors
        if invalid:
            raise ValueError(f"Invalid vectors: {invalid}. Valid options: {valid_vectors}")

        # Prepare the file upload
        try:
            files = {"file": open(file, "rb")}
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file}")

        data = {
            "vectors": ",".join(vectors),
            "mode": mode,
        }
        if webhook_url:
            data["webhook_url"] = webhook_url

        try:
            response = self.session.post(
                f"{self.base_url}/screen/dataset",
                data=data,
                files=files,
            )

            if response.status_code == 401:
                raise ValueError("Invalid API key. Check at https://darkfield.ai/settings")
            elif response.status_code == 402:
                raise ValueError("Usage limit exceeded. Upgrade at https://darkfield.ai/pricing")
            elif response.status_code == 413:
                raise ValueError("File too large. Maximum size depends on your plan.")

            response.raise_for_status()

            result = response.json()
            return Job(
                id=result["id"],
                status=result["status"],
                report_url=result.get("report_url"),
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to submit job: {e}") from e
        finally:
            files["file"].close()

    def get_job(self, job_id: str) -> Job:
        """Get the status of a screening job.

        Args:
            job_id: The job ID returned from scan_dataset()

        Returns:
            Updated Job object with current status
        """
        try:
            response = self.session.get(f"{self.base_url}/jobs/{job_id}")
            response.raise_for_status()

            result = response.json()
            return Job(
                id=result["id"],
                status=result["status"],
                report_url=result.get("report_url"),
                results=result.get("results"),
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get job status: {e}") from e

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> Job:
        """Wait for a job to complete.

        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait

        Returns:
            Completed Job object

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            job = self.get_job(job_id)
            if job.is_complete():
                return job
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    def list_jobs(self, limit: int = 20) -> list[Job]:
        """List recent screening jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of Job objects
        """
        try:
            response = self.session.get(
                f"{self.base_url}/jobs",
                params={"limit": limit},
            )
            response.raise_for_status()

            result = response.json()
            return [
                Job(
                    id=j["id"],
                    status=j["status"],
                    report_url=j.get("report_url"),
                )
                for j in result.get("jobs", [])
            ]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list jobs: {e}") from e

    def get_usage(self) -> dict[str, Any]:
        """Get current usage and quota information.

        Returns:
            Dict with usage stats: samples_screened, samples_remaining, etc.
        """
        try:
            response = self.session.get(f"{self.base_url}/usage")
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get usage: {e}") from e
