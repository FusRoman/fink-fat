# Fink-FAT — Fink Asteroid Tracker

[![CI](https://github.com/FusRoman/fink-fat/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FusRoman/fink-fat/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/FusRoman/fink-fat/branch/main/graph/badge.svg)](https://codecov.io/gh/FusRoman/fink-fat)
[![Docs](https://img.shields.io/badge/docs-rustdoc%20%2B%20KaTeX-blue?logo=rust)](#documentation)
[![Release](https://img.shields.io/github/v/release/FusRoman/fink-fat?include_prereleases&sort=semver)](https://github.com/FusRoman/fink-fat/releases)
[![License: CeCILL-C](https://img.shields.io/badge/license-CeCILL--C-blue)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024%20edition-orange?logo=rust)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![Fink][badge-fink]](https://fink-broker.org)
[![arXiv](https://img.shields.io/badge/arXiv-2305.01123-b31b1b?logo=arxiv)](https://arxiv.org/abs/2305.01123)

**Fink ecosystem:**
[![fink-broker][badge-broker]](https://github.com/astrolabsoftware/fink-broker)
[![fink-client][badge-client]](https://github.com/astrolabsoftware/fink-client)
[![fink-science][badge-science]](https://github.com/astrolabsoftware/fink-science)
[![fink-filters][badge-filters]](https://github.com/astrolabsoftware/fink-filters)
[![fink-fat][badge-fat]](https://github.com/FusRoman/fink-fat)

<!-- Fink badges use the official palette (fink-broker.org/visual-identity): navy #15284F, blue #3C8DFF, orange #F5622E, grey #D5D5D3. The Fink badge embeds the official secondary logo. -->
[badge-fink]: https://img.shields.io/badge/Fink-astronomy%20broker-F5622E?style=flat&labelColor=FFFFFF&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAADAAAABACAYAAABcIPRGAAAS8ElEQVR42sVaeZBdZZX%2FnfN9977uzk5IJ01CYKCTDi0EwyKK6AtiICYEWXwo7gMIVmHpDI4z1oza8wa3mnFGx7EsoQAXZKR4QiIkEAgxPDdACJvQIQuBpAJJmqy9vnfvd86ZP%2B7rTnfW7gDOV9X1vbr39r3f951zfud3FuCvOQoFBwBNJ31w%2BuRZl98xqeXyHY2zLuuYNHPhTdkDbTzSV1L%2FS9%2ByUSrJIRdfKsnEmfPPZm74NbGbbloFwHAcAaHvhm3rlv64%2F7n9%2FptRKNDBN%2FBXGW0MFPXYU%2BbPIeRWEdw405ASu8gUgRgMTV%2FpGL31FKxeHQDYsCXQNPPSnyp7L5xmVxQAG6AO4BTQCGABtLbXQ90jKJHVK%2Bs3dr6w9KX%2BRQNgAJjSXJgYfFhNjOOhaiAmU9tOZJMBhRlSZ2jZvu7%2BVwb%2BtyaNY2de%2FCG4hk8CSSBSNnUAC7xq8OD4s8QEZgDGgANABhAB5ABigHxNVnTwezCQz0FC5yMTKuGVnSg4oJipQT7PKJeDuPRnzHw8RFMCBUG4RiVd5tkvMR%2BdD%2BVINEwD8ArQnp1WR0c2szuLo7qPW1A4MqhzAHkYBfigSQ9McxBDbWUEsn1CpJpA958H7pESE2sStjX06pUbNi%2Bv9p868nmPcjlMnLnwH8n7BSahykyxD9XLtq9%2FYBkATJq5aDnBnU8AlGh0Zi8ASoPUhFA17VOVSgDMAd4AgpFWWZ14I%2FYg8iAlIw1mFAAKBgqo%2Fe6fD7hHCEacwMIXN29etrsGCgoUHMrl0HTypWd6ir5loZow%2B5xo8rXX1j%2BwbNq0Qj0AIrO9BAGgIEUmtdL%2Bmp4qSAORC0aeQPAgeAK8JyKQmRJHbJKuFNd7rYZ61%2BBNhmNEZjElSa%2Fs2rB8CwCqIQj1I48%2BKzcbwYMJKkl5x9pl30Y%2B77d0bwzYksnSiKAkYE9dQ15eLgcAkFRviYTv4RCLRnKrOZ5nmqgR4L14qBnABJD17G5%2FZDMA7B051OxTrkKBUSpJ43OFGyXnzuS0EkhcoiyfA0Aoz1XkH2VkZjeWjQC1RIl2Zq9qHYJCuzYs7wTQCQBTZi3qAXKZNhvgzWwwarlMf69zQJMMf%2B3FfvzKILNU1OmzLmuqmLZRKkIu50X62nasWbY%2Bs4tiAPKcAbw1gRwM2hVruqv2PjvwcK7z2ZqeHvBbRoAXDiDjfodg2UJ2E3CLHp23bSeUoH1KbeRpLIxMJH2h0TV8bwcKDuWakyo3GgAExgmODGy2Y2tT1x6sO7imArsVuEWFFlm%2F8yIDmGEgsreKKziUSjql%2BZJWIlxjkigBFJv8XXt7Kdm3GAAoabYIPoHAILJNmc4P0Ak6xE6G%2FM6A3%2FitdLtmTDcZk2eO2ETve33d0pX5fN73L7oGszZ91kVNBGuGGZhobab%2B7b62NjsidSCAhRxs4LK9udNHSY9rvmQO2C5TS01N0jEc%2FuXuQsGVM0QxKxQckPGalOlkMI8zA8j8EwCA9lLS0nLJmOmzLmo62Jqd7btABjCZwoZ4pqNdf3YCge2fjWLyFJNT%2B9nGl5a9cGWpJPaVc8cAAJVKcuZ1GxkAquzPYMrBJAnB9b4IAMeesuCaPS63piK5dw8g2qFUiADvLMDAb4nuN7Z86DRjXAYVY7LKtt3xN6pXzzoVLvpeurdzdnrt7I6Q2g%2Fqb1n9MwNocvCzyTEItmlsomvegJHpon8CV6eS9P2xxm6HqIUSBlabSYDeAkJaO32y6KsM75xzVBG92e7%2B8x7l%2BJE4ootMuMmxO71ulPtpz7VnX0aAEfTsGqQ8t2HD8uqxp8xrdkwnq9DT2zeu6KipxBA0ZBsqARbyMLg3YQNtjFJJp51caDZEHwkwY02re7bxN1Ea%2F7G6nJ%2FcUwnVFLDeIAkM2kDpdRhzw0QYzVQolOwPAIi17gPkxzGIyjUudUCsojSIihnANPjK0dhAoZ0AWOoqX4CXmMkTm92FriU7t1Z1KqAGgAlGABwMBhciNG49iziqo5CIqf4hkyDPMygA%2FdMhIW7QDyOADQKQHq3yEEolmdJ83iQlfMYM5jQEJf525fr3z1DNPRESJe8oMrMAqCF2DlHyAFI9h7yDmr66c8zWZ2fPnjfKCHM17TVSeTpzdnP1YDBKg86axQnsaB1ZTcRGx3wWrn68kQdMHnr9pSXrOqvdv24wfUdVw9WeuToqBz8qjnylJ12GY6o%2FPiYnFysIBnoKq1enrye5sxD5iU7TNR3HdW8%2BBKWAG3Ql88TGoKODT0K5LCeckK8D0fWmYsxCO6qjbrIvtbRMiMPsGGlb13h%2Fd9rjT636tNBZce%2Btv%2B35i3N3n3q8czSHVODBDwMg0%2FhCpjpj41UolwPyeX8wozwARr32s9ERGnE%2B71Aui0bj55mPTiYL5tLkhfDq4id27XnH7WNz3urq0gnJLny%2F%2FpfPXAdgQ%2B0LNMX4InM5r9LXwxyvBGAO6QWkjgLzqsN9VvaH0aFsdASSyMiYJeDPZ4bpaG8af89%2B1Dq6j%2BKP7U2V9lRYIrjPbvnUnPdYoeAemN%2BcI8BS0AImNjJ%2Betuaezc1nnbBSYA7Q1LpzWl4vBYLHN4wbTAbHbEStTFQlEmti5othA8CICdhT%2B%2FGE%2B%2Fa%2BsxrV4%2BLUL%2BnKsErUDeao4T0Qrqj9JgZ9IR3LJjSJ3YuTMlgywFQmo66IIobIqr0PrZlw7LXDob%2FQ2wgcyKZBAYzoeGrTxaMQOzT5KKYwBClexj%2FU32jgi92Vw1BQSmR29udVlygX6KGpYn4C8jlxklIEoDvB2DOsABEJs6vOBT%2BHwzoazDqB7FRG7bxonl%2BjsxdlXFixZ7d0Q%2FlhtPOGO101u5UNDFYFDlUlFae8IvVL7fl894AqPoC2BmgL3as%2Fc1fpjQXJoFwvkkvkVUeHRwrHMoGhqBQIIaOSAYFBmCN7M8jx80Gg6q8mHbc8%2FxzO0b9PeAshWlFlasi1IXoJwDwr4%2BWZdqMhVPVJR8wA8HcgwBIXbjQ%2Bfpx0LBxnIYnB8cKh3Jkg3fHrAoaUDcaLu8BiK8yInPk0Kt0q%2F3ntPoq2aV7E6ZgjiL2vLMP633a83A%2Bn%2FdEsKqLLoOrHyOSBBO6B4AppEDkzJFfuWHD8motq2HDViGHABpBJg%2BlkhzTPH%2BskC4yM4KmSc%2FWsXc898yUD%2Be8H10RkV5zxhRDEd16aqk9%2BXh3NwFGLPYJZ84Y8tyOlxc%2FM7U5Pw1mF5gmBOiS4SzgwHhgJGw0My6KnbuQnW8EEUzpT%2Bi6Y%2Bf2Xvf5SlBUzEzV3PYk6drN1V8CoOtXr06Pm3Hp6UZ8tlJKzngJAFM34QoX50abVF8fp%2Fb7Gn0eNq8xAjiMhI02ZtgfiD5qgDEROiXcbp9%2FT6PBn7c7iCUG5NhRn%2BiyC%2B965vX58%2BfHWfSln9KYnGjSJ77yKwAIZleB2MzowbVr7%2Bs6kvoAQNifjbIQYMNyZIRSSY6bdelEGF1AakSSViov1y9esVOviiNyiaoEJe5RIEX9zQDw4DnnpJNa86ON9EpnzkjdY2%2B8uPzlppOvPAOMs1SEzPiekUfe%2FTDKIdvKMIlbCsmzcxOIGCZuFVDq3lWhz%2FSlhiCROZfjvSmeferUP%2F4un897KhaVZNyHHcXT1AIZ7DYA0Kjv0%2BwjJxI21fVGjw5XfQZsoN%2BRCevw2GimPnCwRQAZw6M7wS%2FsE2c2KcXv7AqGVEEGB1F3a7EIndv4RuZgjK8liowk3RSP3rJkUmthtIA%2BYgQw8YNbtpT6DkXejlTQYGcOZDQs9GlqurjBwPMMIBPp6%2BmesOzurjGX%2B8hRBZYqsd9Zld2O6%2B4CQMVSezJ1xoWnG%2Bv7jIyY%2BM6tq1f3Oq1cwZSbamIWsd15JOd1ODbKTh3oiJ44ywwkY%2BNzxPFUAmCmT2Ln7V1bJb2qNyjSwAqOUIEtvXzJyp39xhsw6m%2BJc05C6I01d2vGQOgaZjbW9IXz2qPHsoMtDSuVKfsb8bDYaD4rNBDpQjIPBiEJes8rXzp9PJTP7ksEQcn3akCFK7cDwDnnLE%2BnT184QYmvApERZMXmdaVXprRccRYI55oFAuTOEkpyOO5z4OD9gnpOYXwE2ymXBQA7DRewGmCCPXvcvcs3HnOh9%2FVxn7pEvXedqa7ZsWf0H%2FL5vC8WoZW47qPw1AitkpnenMk4vZ64wYkkPcz0q2FR50HD7%2B%2FI3BEJRBsDsIktl8wQ5lMBg4muw67FWzp7cGUaBImpGghVdf9bLJfD3P5ckUuuI2Izsb%2FsOL3%2B4UmtC6Yo2UdAwbzyI1vbl26uvf%2BognIjgJUi2OFsoEadPfBeZufBhArTQ3Y3XB9kbk8ISIzizqqmaeJLAFAsl8OU5r7zQPEcmBGBbkepJCS4ljg3HhrITG8dQs2HOQ6wAckC6yPCp5rNA7I0%2FF7xv7795%2B86VSM3sU81MMXcp%2B7Jb6xatTZL4gLMcj05NtPqjrpc%2BPm0aYV6Mv85YzWTdO32cU0P1ah5GKH%2FGpqddiqHY6OEUklaWwuxEb1HAUBCJ9af9vj63lEL1CJL1AWBQ6q8GAA9Ws5oc4BfpERE5n6z%2BS%2FLdlfrQ8FcPN0MxORuw%2Bpb0pEZ7yGy0w5yGDaawWdnH51O8Mdnz8lTQDHpsmhBNRAF5WiPhCCQ%2B2tqaYH8DRblRltIxdT%2FCAAxpzcqq0HCbqTu54PAYUTjADZqdBgVrMFnEqXvZXbsQEjUrbRCa1xNwjt7QmLq2KWiz%2F7HqlXr%2BtVHEfqIY3it%2Fv6N9aVnJ8%2B6dCFxfDqrI1JbvH3j4o7hELfhGLEX%2BEFWYAfLPIBI5vZH0l3V3PIbd058J0VudEWkGlmc01TuA2At3S1URplob%2FJfMqF6Y7DcT7OErH7d2JtJKrHpD%2FcrpTLy%2BSMb8mtdDlPzFDr2HTkZ4FntUPUBAkrSdObFDdpFZyoBpmlvurnnxY5pDV%2BujyIErbqukKoY7geApoubBKvbaPv2Ys%2BkCQuuIfYrjp9x0fsSh3eZCQjh4dfWL3uuP6vRn68dph8I2ADYrEuSgYIeAV5rRb6D639JrFtald202o5fBJZXq%2FrBvAkD5HwaQvvMxx55HgAVi8UBNHjjpQfuBQDfcsU3LXMeYPH%2Fnt1up9ombPLMeSeCxy4IlCizEcQBLgUkApwAktVOxYwpNAijdwZMABCRGbyygfUgQJrvIJQBSO4sxL5fwZ4yA11yjs0JIaDeRUhAvy0Cms%2FnfXkAEgsMlHTKzEvnG7v3w4JB5bFtG%2B4vD5x%2BoeBQgpI2fJdy0ZVOatkpZwDiQXP2ZSIDPIE0gpoAYDIYvFMvUDDYbIhHrOG%2FkZxL5mFEqMCt%2Btzc%2FFThaJJIKkzmgvDy7PHGQXrYakCBxeQmQlKLXPm7%2B8qwbYxSUSa3XHSigBYi7alY1qpBQxoyzNViFa1dC4Cpo4HnIOzJNbCLI%2Bacg9GogdMvlaQNbSzMZxtUYKl29%2FmnNlXdHCEPQcQ9qXb1VfTJLBapBSP5vAOK2jgjvdI5d6YRYCpPbl%2B7ZGmtGCL93let4esujkcx%2BTrn4jrHLsecy%2B2bo5xjX5vj7M9FnjhyzJFzxA0%2B4eolJKwedeaAjsHo85OWx6cb5U5kdg5p6MXm0iuVyfM%2BGbGBkCND%2Bvxvn3%2BoY1%2BiL2sjmDw7HhXS5FtkZoSIlJIiQJa1IAz0QJA4%2BRWJ%2Fw1IAAsEOAiqcANzlvbtt3WB1O7VNEuD%2Bp0vLr0fh9B%2FVf%2F%2BOIrrUg2PmUt%2FAQC9gefEjhDDEMSeyw59riuXEbKMdTGgsuirHOX%2BRlSMpPL4jrVLl9VaEAY7Ltu15r5H3mx5zg%2FpmSu1GlAcgLXY4nVI7d071i9%2BYqBGZe6EVD2YDCl49RDWWi7K1JMWzkjJ%2FQMkVWbH4PC1QS0IB1Y3C292Awdv0lMAeG3D4sf3RREFam2FC9I7UaBZB5roqwMGXOuRSCL%2FfWauAxEspEvfWLd8Za0Me5DvlOTATY1s8JHT6P0SKon3YbyomyBqSEJIUuNXAWDjxo2MUkmmzlhwOTm%2FMJgGCFU9%2FFf2odLbM46wgaJmsWobAKALneMTpKMSY1SUtnV2dr8OgFafdJJOas2PTin3A6io48hD7b%2B3rrv3pazSXtT%2Fpw0MHdUUoypGrqIOFaVdmzbNTdBaiFAqCYUx3zHvjlcygiQbqL7z3%2FpryG9nQ6cfWRtKHQeLwUwwQwwUFe1IJrfMv1qd%2BwKkmhJHEWu4ZtvzK3qA8e5ow8W3ZQNQ6gGLqhoZ%2FMzGlkVfZnOjlK2oaknEdbFK%2BM629ff9LuvMKgW8zYNG8Jwd865PjHVdXS%2BT8bECg%2BMYgEIlDexy3qS6omPtfRdRrYL5Zvn%2BW2kDBoB3%2FfnOTjK8pOyMzYJqJaikFXZ1HiJrci73MdqXJnnbFz8yI641X7NE33cgIpAnOM%2Fe10GTJ2OtztvSXtoFtNHbrfdDQsxhP9nebgC4e9ea9jHjZpmxn02gbojcZm7PZ7atXbFzUL%2F0X238H83VZOfn%2BDEFAAAAAElFTkSuQmCC
[badge-broker]: https://img.shields.io/badge/fink--broker-alert%20processing-15284F?style=flat&labelColor=F5622E&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPGNpcmNsZSBjeD0iNyIgY3k9IjEyIiByPSIzIiBmaWxsPSIjRkZGRkZGIi8%2BPHBhdGggZD0iTTEgMTJoM00xMCAxMmg0TTEwIDEwLjUgTDE0IDVoOE0xMCAxMy41IEwxNCAxOWg4TTE0IDEyaDgiIHN0cm9rZT0iI0ZGRkZGRiIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48Y2lyY2xlIGN4PSIyMSIgY3k9IjUiIHI9IjEuOCIgZmlsbD0iIzE1Mjg0RiIvPjxjaXJjbGUgY3g9IjIxIiBjeT0iMTIiIHI9IjEuOCIgZmlsbD0iIzE1Mjg0RiIvPjxjaXJjbGUgY3g9IjIxIiBjeT0iMTkiIHI9IjEuOCIgZmlsbD0iIzE1Mjg0RiIvPjwvc3ZnPg%3D%3D
[badge-client]: https://img.shields.io/badge/fink--client-alert%20client-3C8DFF?style=flat&labelColor=15284F&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZD0iTTQgNiBhMTAgMTAgMCAwIDAgMTQgMTQgeiIgZmlsbD0iI0ZGRkZGRiIvPjxwYXRoIGQ9Ik0xMSAxMyBMMTcgNyIgc3Ryb2tlPSIjRkZGRkZGIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxjaXJjbGUgY3g9IjE4IiBjeT0iNiIgcj0iMiIgZmlsbD0iIzNDOERGRiIvPjxwYXRoIGQ9Ik04IDIwaDgiIHN0cm9rZT0iI0ZGRkZGRiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L3N2Zz4%3D
[badge-science]: https://img.shields.io/badge/fink--science-science%20modules-15284F?style=flat&labelColor=3C8DFF&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZD0iTTkgMmg2TTEwIDJ2N0w0IDIwYTEuNSAxLjUgMCAwIDAgMS4zIDJoMTMuNEExLjUgMS41IDAgMCAwIDIwIDIwTDE0IDlWMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjRkZGRkZGIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48cGF0aCBkPSJNNyAxNWgxMGwyLjYgNUg0LjR6IiBmaWxsPSIjRkZGRkZGIi8%2BPGNpcmNsZSBjeD0iMTEiIGN5PSIxMS41IiByPSIxIiBmaWxsPSIjRkZGRkZGIi8%2BPC9zdmc%2B
[badge-filters]: https://img.shields.io/badge/fink--filters-alert%20filters-15284F?style=flat&labelColor=D5D5D3&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZD0iTTIgM2gyMGwtOCA5djdsLTQgM3YtMTB6IiBmaWxsPSIjMTUyODRGIi8%2BPGNpcmNsZSBjeD0iMTIiIGN5PSI3IiByPSIyIiBmaWxsPSIjRjU2MjJFIi8%2BPC9zdmc%2B
[badge-fat]: https://img.shields.io/badge/fink--fat-asteroid%20tracker-3C8DFF?style=flat&labelColor=F5622E&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPGVsbGlwc2UgY3g9IjEyIiBjeT0iMTIiIHJ4PSIxMSIgcnk9IjYiIHRyYW5zZm9ybT0icm90YXRlKC0yMCAxMiAxMikiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0ZGRkZGRiIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyLjYiIGZpbGw9IiNGRkZGRkYiLz48Y2lyY2xlIGN4PSIyMC4zIiBjeT0iNy42IiByPSIyLjQiIGZpbGw9IiNGNTYyMkUiIHN0cm9rZT0iI0ZGRkZGRiIgc3Ryb2tlLXdpZHRoPSIxIi8%2BPC9zdmc%2B

Fink-FAT finds **new Solar System objects** in the photometric alert streams of
large time-domain surveys (ZTF, Vera C. Rubin / LSST). It links
detections night after night into candidate trajectories, keeps several
association hypotheses alive while the data is ambiguous, and turns the
surviving candidates into orbits and MPC-ready ADES submissions.

The project started as a Python package used inside the
[Fink broker](https://fink-broker.org) (see the
[paper](https://arxiv.org/abs/2305.01123)). Version 1.0 is a full rewrite in
Rust.

## How it works

Fink-FAT treats linking as a **tracking problem**, not as a combinatorial
search over all possible combinations of detections:

1. **Intra-night seeding.** Within a night, detections of the same moving
   object are linked into tracklets. A tracklet only constrains the sky
   position and angular rate. Range `ρ` and range-rate `ρ̇` are unknown.
2. **Admissible-region hypotheses.** Each tracklet seeds a *Kalman filter bank*:
   a set of topocentric Kalman filters, one per `(ρ, ρ̇)` node of the
   admissible region. Nodes are weighted by dynamical population priors
   (NEOs, main belt, Hildas, Trojans, Centaurs, Kuiper belt, scattered and
   detached objects).
3. **Night-to-night multi-hypothesis tracking.** Every lineage is propagated to
   each new exposure. Detections inside its predicted search ellipse spawn
   **branches**, which are scored by a log-likelihood ratio against the local
   clutter density, then pruned (top-B / N-scan). Range ambiguity collapses as
   the arc grows.
4. **Discovery and archiving.** Detections that no existing lineage claims seed
   new lineages. Lineages that stop receiving detections are archived, not
   lost: their association history is still a result.
5. **Orbit fitting and submission.** Candidate tracks are fitted (via
   [`outfit`](https://crates.io/crates/outfit)), cross-matched against known
   objects (SkyBoT, CND), ranked by a submission-quality tier and exported as
   ADES XML to the Minor Planet Center.

Steps 1–4 run in the engine. The
[engine README](crates/fink-fat-engine/README.md) describes the algorithms in
detail: state vector, filter equations, scoring, pruning and fragment
merging.

A few principles guide the design:

- **Completeness for unusual objects first.** Fink-FAT aims to find NEOs,
  Centaurs and distant objects, not only main-belt asteroids. Tuning choices
  are judged on metrics split by population, not on aggregate recall alone.
- **Incremental by construction.** Night 0 and night N use the same call
  (`BranchCollection::advance_one_night`). State is a single on-disk snapshot,
  so a nightly cron job and a multi-year replay run the same code.
- **Pure logic, thin I/O.** Algorithms are pure functions over plain data
  types. Postgres, HTTP and filesystem access live at the edges (CLI,
  explorer). For example, `fink-fat-ades` holds no I/O at all, so the CLI and
  the web app always agree on what a valid submission is.
- **Validated, unit-aware configuration.** One YAML file drives the whole
  engine. Unknown keys are rejected. Physical quantities are written with units
  (`"35 arcmin/day"`, `"0.02 au"`). Every validation error is reported at once.

## Workspace layout

| Crate | Path | Role |
| --- | --- | --- |
| `fink-fat` | [`src/`](src/) | Main CLI: `track`, `convert`, `submit`. |
| `fink-fat-engine` | [`crates/fink-fat-engine/`](crates/fink-fat-engine/README.md) | Core library: configuration, seeding, space-time bucketing, topocentric Kalman filters, branching / MHT, archive, snapshot. |
| `fink-fat-ades` | [`crates/fink-fat-ades/`](crates/fink-fat-ades/) | Pure ADES / MPC logic: document building, local `submit.xsd` validation, submission and status API parsing, quality tiers. |
| `fink-fat-eval` | [`crates/fink-fat-eval/`](crates/fink-fat-eval/) | Evaluation binaries run against labelled datasets (tracking recall/purity, seeding, KF calibration, MOT contamination). |
| `fink-fat-explorer` | [`crates/fink-fat-explorer/`](crates/fink-fat-explorer/) | Dioxus fullstack web app on Postgres: population maps, 3D orbits, lineage pages, orbit fitting, cross-matching, MPC submission dashboard. |

Other notable paths:

| Path | Content |
| --- | --- |
| [`tests/`](tests/) | CLI-level integration tests and small ZTF / LSST night samples. |
| [`test_exp/`](test_exp/) | Python scripts for data preparation (alert Parquet files, ground truth, astrometric error calibration). |
| `engine_config.best.yaml`, `engine_config_lsst.best.yaml` | Reference engine configurations for ZTF and LSST cadence. |
| [`dockerfile`](dockerfile), [`docker-compose.yml`](docker-compose.yml) | Postgres, pgAdmin and the explorer (`dev` / `prod` profiles). |
| [`katex-header.html`](katex-header.html) | KaTeX header so rustdoc renders the maths. |

## Requirements

- Rust stable with 2024 edition support (≥ 1.85).
- `libfontconfig1-dev` (used by the plotting code in `fink-fat-eval`).
- Network access on first run: the Kalman context downloads the JPL ephemeris
  and UT1 data, and the MPC observatory list.
- Optional: PostgreSQL (or `docker compose`) for `convert --format sql`,
  `submit` and the explorer.

## Quick start

```bash
cargo build --release
./target/release/fink-fat --help
```

### Track asteroids

```bash
fink-fat track \
  --alerts /path/to/alerts.parquet \
  --config engine_config.best.yaml \
  --progress --logs
```

`--alerts` is a Parquet file holding one night, or several nights with
distinct `night_id`s. Nights are processed in chronological order. The run
loads the `BranchCollection` snapshot from the config's `storage_path` if there
is one, advances it through every night, then writes it back. Closed
trajectories go to an append-only archive log next to the snapshot:

```text
<storage_path>/
├── branch_collection.rkyv          # live lineages (zero-copy rkyv snapshot)
├── archived_trajectories.rkyvlog   # append-only log of closed trajectories
└── fink_fat.<date>.log             # with --logs; rotated daily
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--snapshot-every N` | In batch mode, also checkpoint the snapshot every N nights. |
| `--log-target TARGET=LEVEL` | Per-target log level (repeatable). `--list-log-targets` lists them. |
| `--log-retention-days N` | Override how many daily log files are kept. |

Any configuration key can be overridden from the environment with the
`FINK_FAT__` prefix and `__` as the nesting separator.

### Export results

```bash
# Flat Parquet tables
fink-fat convert --config engine_config.best.yaml --format parquet

# Postgres (feeds fink-fat-explorer and `submit`)
fink-fat convert --config engine_config.best.yaml --format sql \
  --database-url postgres://finkfat:finkfat@localhost:5432/finkfat_test \
  --path-observation /path/to/alerts.parquet
```

The Parquet export writes one `<table>.parquet` file per table into
`storage_path`.

### Submit to the Minor Planet Center

```bash
fink-fat submit \
  --csv eligible_lineages.csv \
  --database-url postgres://finkfat:finkfat@localhost:5432/finkfat_test \
  --submitter-config submission.yaml \
  --dry-run
```

`submit` skips lineages that were already submitted, checks that each lineage
is eligible (by quality tier), builds the ADES document and validates it
locally. Then it sends the document and records the outcome in the
`mpc_submissions` table. `--endpoint test` is the default and targets MPC's
test endpoint. Only `--endpoint production` sends a real submission.
`--dry-run` stops before anything leaves the machine.

### Explore results in the browser

```bash
docker compose --profile dev up -d            # db + pgAdmin + explorer (hot reload) on :8080
docker compose --profile prod up -d --build   # optimised build on :8081
```

See [`crates/fink-fat-explorer/README.md`](crates/fink-fat-explorer/README.md)
for the build targets, runtime configuration and data loading.

## Evaluation

`fink-fat-eval` runs the engine on labelled data (alerts with their
ground-truth object identity) and reports where reconstruction succeeds or
fails. Each tool is a separate binary:

| Binary | Purpose |
| --- | --- |
| `tracking_analysis` | Full night-after-night tracking: recall, purity and fragmentation, broken down by population. |
| `seed_to_bank_analysis` | Intra-night seeding in isolation (every night treated as night 0). |
| `kalman_full_traj_eval` / `kalman_single_traj_eval` | Kalman bank behaviour on ground-truth trajectories (coverage, NEES/RMSE). |
| `kf_calibrate` | Automatic search for search-region / gating / process-noise parameters. |
| `mot_analysis` | Contamination of the predicted search regions by unrelated detections. |

```bash
cargo run --profile evaluation-fast -p fink-fat-eval --bin tracking_analysis -- \
  --alerts /path/to/sso_dataset_eval.parquet \
  --config engine_config.best.yaml \
  --output-result eval_out/
```

The `evaluation` profile keeps debug assertions and overflow checks to catch
numerical bugs. `evaluation-fast` drops them for full-dataset sweeps.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace
```

Tests use synthetic or bundled data only. The `sql_convert` integration test
needs a Postgres instance given by `DATABASE_URL`; `docker compose up -d db`
provides one with the same credentials as CI.

A `husky-rs` pre-commit hook runs fmt, clippy, check and the doc build. CI runs
the same jobs, plus the Postgres integration test and engine coverage
(Codecov). Contributor conventions (error handling, collections, doc comment
format, tests) are listed in [`AGENTS.md`](AGENTS.md).

## Documentation

For an overview of the algorithms, start with the
[engine README](crates/fink-fat-engine/README.md).

The API reference is generated with rustdoc. Module-level docs explain both
the algorithms and the reasons behind the design choices. KaTeX renders the
equations:

```bash
RUSTDOCFLAGS="--html-in-header $(pwd)/katex-header.html" cargo doc --workspace --no-deps --open
```

Good entry points are `fink_fat_engine::topocentric_kf::kalman_bank` (the
filter bank), `fink_fat_engine::topocentric_kf::branching` (multi-hypothesis
tracking) and `fink_fat_engine::engine_config` (every configuration key and its
units).

## License

Fink-FAT is distributed under the
[CeCILL-C](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html) free
software license, a French LGPL-style license compatible with EU law. See
[LICENSE](LICENSE).

## AI Disclosure

D3 — daliascale.org

D3: Human and AI act as creative partners. AI shapes ideas or output in meaningful ways, but humans retain authorial control.