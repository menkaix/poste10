import email
import imaplib
import ssl
from dataclasses import dataclass
from email.header import decode_header


@dataclass
class EmailMessage:
    uid: str
    subject: str
    sender: str
    date: str
    body: str


class ImapEmailReader:
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def _connect(self) -> imaplib.IMAP4_SSL:
        ctx = ssl.create_default_context()
        mail = imaplib.IMAP4_SSL(self.host, self.port, ssl_context=ctx)
        mail.login(self.username, self.password)
        return mail

    def fetch_unread(self, n: int) -> list[EmailMessage]:
        mail = self._connect()
        try:
            mail.select("INBOX")
            _, data = mail.uid("search", None, "UNSEEN")
            uids = data[0].split()
            uids = uids[-n:]  # derniers n emails non lus

            messages = []
            for uid in uids:
                _, raw = mail.uid("fetch", uid, "(RFC822)")
                msg = email.message_from_bytes(raw[0][1])
                messages.append(
                    EmailMessage(
                        uid=uid.decode(),
                        subject=_decode_header_value(msg.get("Subject", "")),
                        sender=_decode_header_value(msg.get("From", "")),
                        date=msg.get("Date", ""),
                        body=_extract_body(msg),
                    )
                )
            return messages
        finally:
            mail.logout()

    def mark_as_read(self, uid: str) -> None:
        mail = self._connect()
        try:
            mail.select("INBOX")
            mail.uid("store", uid.encode(), "+FLAGS", "\\Seen")
        finally:
            mail.logout()


def _decode_header_value(value: str) -> str:
    parts = decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def _extract_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode(
                msg.get_content_charset() or "utf-8", errors="replace"
            )
    return ""
