"""Human-readable alert texts for the system's artificial (convention) calls.

`convention_alert` takes a classified call (auction.py CallInfo) and returns
the alert the bid requires -- or None for natural, non-alertable calls. When
a parsed system file is given (system_file.SystemSpec, the conventions text
file the website serves its alerts from) its alert texts win; the built-in
strings below are the fallback and cover exactly the conventions the rules
reward (rules.convention_kind): XYZ 2C/2D, RKCB asks and step responses,
splinters, Smolen, fourth-suit GF and support cuebids of the opponents' suit.

rl/analyze.py stamps the text into the dumped self-play tree ("conv" field
of each node) for debugging; the live website no longer reads that tree --
the Go server generates its bid alerts straight from the system file
(system_writer/system_file.go + alerts.go).
"""

from .auction import DENOM_N, MAJORS, CallInfo

_SYM = ("♣", "♦", "♥", "♠")        # project denom order C,D,H,S

# step (1..4 = 5C..5S) -> what the RKCB response shows
RKCB_STEPS = {
    1: "0 or 3 keycards",
    2: "1 or 4 keycards",
    3: "2 keycards without the trump queen",
    4: "2 keycards with the trump queen",
}


def _suit(denom: int) -> str:
    return _SYM[denom] if 0 <= denom < DENOM_N else "?"


def convention_alert(info: CallInfo, spec=None) -> str | None:
    """Alert text for a classified call, or None if it is not a convention.

    `spec` is an optional parsed system file (system_file.SystemSpec): when
    given, its sections drive both the call -> convention resolution and the
    alert text, so the website and the training-side dumps speak from the
    same file. Without it the built-in texts below are used.
    """
    if spec is not None:
        return spec.alert_for(info)
    if info.kind != "bid":
        return None
    if info.xyz_2c:
        return "XYZ: 2♣ relay — any invite, or a diamond signoff"
    if info.xyz_2d:
        return "XYZ: 2♦ artificial game force"
    if info.smolen:
        other = MAJORS[0] if info.denom == MAJORS[1] else MAJORS[1]
        return (f"Smolen: 4 {_suit(info.denom)} and 5+ {_suit(other)}, "
                f"game force")
    if info.rkcb_ask:
        return (f"RKCB for {_suit(info.rkcb_trump)}: keycard ask — 5♣ 0/3, "
                f"5♦ 1/4, 5♥ 2 without / 5♠ 2 with the trump queen")
    if info.rkcb_response:
        what = RKCB_STEPS.get(info.rkcb_step, "an unspecified keycard count")
        return f"RKCB response for {_suit(info.rkcb_trump)}: {what}"
    if info.splinter:
        return (f"Splinter: singleton/void {_suit(info.denom)}, fit for "
                f"partner's {_suit(info.partner_last_suit)}, slam interest")
    if info.fourth_suit:
        return f"Fourth suit {_suit(info.denom)}: artificial, game force"
    if info.cuebid and info.partner_last_suit >= 0:
        return (f"Cuebid of their {_suit(info.denom)}: invitational or "
                f"better support for partner's "
                f"{_suit(info.partner_last_suit)}")
    return None
