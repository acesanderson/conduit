def get_logo() -> str:
    """
    Return the app logo as Rich markup so it can be printed safely via EnhancedInput.show_message.
    """
    return (
        "[blue]\n"
        " ██████╗ ██████╗ ███╗   ██╗██████╗ ██╗   ██╗██╗████████╗\n"
        "██╔════╝██╔═══██╗████╗  ██║██╔══██╗██║   ██║██║╚══██╔══╝\n"
        "██║     ██║   ██║██╔██╗ ██║██║  ██║██║   ██║██║   ██║   \n"
        "██║     ██║   ██║██║╚██╗██║██║  ██║██║   ██║██║   ██║   \n"
        "╚██████╗╚██████╔╝██║ ╚████║██████╔╝╚██████╔╝██║   ██║   \n"
        " ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝   ╚═╝   \n"
        "[/blue]"
    )


def print_logo() -> None:
    """
    Backwards-compatible: print logo directly.
    Prefer get_logo() + input_interface.show_message(get_logo()) in enhanced mode.
    """
    print(get_logo().replace("[blue]", "\033[94m").replace("[/blue]", "\033[0m"))
