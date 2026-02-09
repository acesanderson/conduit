from random import choice, randint, choices


class UserAgentRotator:
    def __init__(self):
        # Version ranges (easier to keep current)
        self.chrome_versions = list(range(120, 133))
        self.firefox_versions = list(range(122, 126))
        self.safari_versions = ["17.2", "17.3", "17.4", "17.5", "17.6"]
        self.edge_versions = list(range(120, 133))

        self.os_list = [
            ("Windows NT 10.0", "Win64; x64"),
            ("Windows NT 10.0", "WOW64"),
            ("Macintosh", "Intel Mac OS X 13_6_3"),
            ("Macintosh", "Intel Mac OS X 14_2_1"),
            ("X11", "Linux x86_64"),
            ("X11", "Ubuntu; Linux x86_64"),
        ]

    def generate_chrome_ua(self) -> str:
        os_type, os_info = choice(self.os_list)
        chrome_version = choice(self.chrome_versions)
        return (
            f"Mozilla/5.0 ({os_type}; {os_info}) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{chrome_version}.0.0.0 Safari/537.36"
        )

    def generate_firefox_ua(self) -> str:
        os_type, os_info = choice(self.os_list)
        firefox_version = choice(self.firefox_versions)
        return (
            f"Mozilla/5.0 ({os_type}; {os_info}; rv:{firefox_version}.0) "
            f"Gecko/20100101 Firefox/{firefox_version}.0"
        )

    def generate_safari_ua(self) -> str:
        version = choice(self.safari_versions)
        return (
            f"Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) "
            f"AppleWebKit/605.1.15 (KHTML, like Gecko) "
            f"Version/{version} Safari/605.1.15"
        )

    def generate_edge_ua(self) -> str:
        os_type, os_info = choice(self.os_list)
        edge_version = choice(self.edge_versions)
        build = randint(1000, 2000)
        return (
            f"Mozilla/5.0 ({os_type}; {os_info}) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{edge_version}.0.0.0 Safari/537.36 Edg/{edge_version}.0.{build}.0"
        )

    def get_random_headers(self) -> dict[str, str]:
        """Generate headers with weighted browser distribution."""
        # 70% Chrome, 15% Firefox, 10% Safari, 5% Edge
        browser = choices(
            [
                self.generate_chrome_ua,
                self.generate_firefox_ua,
                self.generate_safari_ua,
                self.generate_edge_ua,
            ],
            weights=[70, 15, 10, 5],
            k=1,
        )[0]

        user_agent = browser()
        is_firefox = "Firefox/" in user_agent

        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Browser-specific headers
        if not is_firefox and choice([True, False]):
            headers["Sec-Fetch-Dest"] = "document"
            headers["Sec-Fetch-Mode"] = "navigate"
            headers["Sec-Fetch-Site"] = "none"

        if choice([True, False]):
            headers["DNT"] = "1"

        return headers
