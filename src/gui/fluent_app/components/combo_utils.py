"""Small ComboBox helpers shared by the settings pages."""


def reset_items(combo, items, index=None, text=None):
    """Replace a ComboBox's items **without emitting change signals**.

    ``clear()`` + ``addItems()`` makes Qt select item 0 and emit ``currentIndexChanged`` /
    ``currentTextChanged``, which the pages' handlers turn into a config write — so merely
    translating a page or refreshing a device list used to overwrite a saved setting with the
    first item until the next load put it back. The previous selection is restored (by ``text``
    if given and present, else by ``index``, else the current index) and nothing is written.
    """
    items = list(items)
    if index is None:
        index = combo.currentIndex()
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(items)
        if text is not None and text in items:
            combo.setCurrentIndex(items.index(text))
        elif 0 <= index < len(items):
            combo.setCurrentIndex(index)
    finally:
        combo.blockSignals(False)
