import typing as t


class Formattable(t.Protocol):
    def rich_format(self) -> RenderableType: ...

    @staticmethod
    def rich_format_many(*items: t.Any) -> RenderableType: ...
