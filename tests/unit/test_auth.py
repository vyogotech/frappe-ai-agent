import jwt
import pytest

from ai_agent.middleware.auth import create_token, verify_token


class TestJWTAuth:
    SECRET = "test-secret-key"
    ALGORITHM = "HS256"

    def test_create_and_verify(self):
        token = create_token(
            user="admin@test.com",
            session_id="sess-123",
            site="site1.local",
            secret=self.SECRET,
            algorithm=self.ALGORITHM,
            expiry_hours=24,
        )
        payload = verify_token(token, secret=self.SECRET, algorithm=self.ALGORITHM)
        assert payload["sub"] == "admin@test.com"
        assert payload["sid"] == "sess-123"
        assert payload["site"] == "site1.local"

    def test_invalid_token(self):
        with pytest.raises(jwt.InvalidTokenError):
            verify_token("invalid.token.here", secret=self.SECRET, algorithm=self.ALGORITHM)

    def test_wrong_secret(self):
        token = create_token(
            user="admin@test.com",
            session_id="sess-123",
            site="site1.local",
            secret=self.SECRET,
            algorithm=self.ALGORITHM,
            expiry_hours=24,
        )
        with pytest.raises(jwt.InvalidSignatureError):
            verify_token(token, secret="wrong-secret", algorithm=self.ALGORITHM)
