import shutil
import subprocess

import google.auth.exceptions
import google.auth.transport.requests
import google.oauth2.id_token


def _gcloud_identity_token() -> str:
    """Obtient un identity token via la CLI gcloud (pour les credentials utilisateur ADC)."""
    gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if gcloud is None:
        raise RuntimeError("gcloud CLI introuvable dans PATH")
    result = subprocess.run(
        [gcloud, "auth", "print-identity-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_identity_token(audience: str) -> str:
    """Retourne un identity token Google pour l'audience donnée (URL Cloud Run).

    Stratégie :
    1. Service account via GOOGLE_APPLICATION_CREDENTIALS ou metadata server.
    2. Fallback sur `gcloud auth print-identity-token` (credentials utilisateur ADC).
    """
    try:
        auth_request = google.auth.transport.requests.Request()
        return google.oauth2.id_token.fetch_id_token(auth_request, audience)
    except google.auth.exceptions.DefaultCredentialsError:
        return _gcloud_identity_token()
