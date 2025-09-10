import importlib
import typing as t

if t.TYPE_CHECKING:
    from types import ModuleType


class LazyImportError(ImportError):
    def __init__(self, module_name: str, extras: str, package_name: str | None = None) -> None:
        super().__init__(
            f"Module '{module_name}' is not installed. Please install it with `pip install {package_name or module_name}` or `dreadnode[{extras}]` extras."
        )


class LazyImport:
    def __init__(self, module_name: str, extras: str, package_name: str | None = None) -> None:
        self._name = module_name
        self._extras = extras
        self._mod: ModuleType | None = None
        self.package_name = package_name

    def _load(self) -> t.Any:
        if self._mod is None:
            try:
                self._mod = importlib.import_module(self._name)
            except ModuleNotFoundError as e:
                if e.name == self._name:
                    raise LazyImportError(
                        self._name, self._extras, package_name=self.package_name
                    ) from None
                raise
        return self._mod

    def __getattr__(self, item: str) -> t.Any:
        return getattr(self._load(), item)


class LazyAttr:
    def __init__(
        self, module_name: str, attr: str, extras: str, package_name: str | None = None
    ) -> None:
        self._module_name = module_name
        self._attr = attr
        self._extras = extras
        self._value = None
        self.package_name = package_name

    def _load(self) -> t.Any:
        if self._value is None:
            try:
                mod = importlib.import_module(self._module_name)
                self._value = getattr(mod, self._attr)
            except ModuleNotFoundError:
                raise LazyImportError(
                    self._module_name, self._extras, package_name=self.package_name
                ) from None
        return self._value

    def __getattr__(self, item: str) -> t.Any:
        return getattr(self._load(), item)

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._load()(*args, **kwargs)

    def __repr__(self) -> str:
        status = "loaded" if self._value is not None else "unloaded"
        return f"<LazyAttr {self._module_name}.{self._attr} ({status})>"

    def __dir__(self) -> list[str]:
        try:
            return sorted(set(dir(self._load())))
        except LazyImportError:
            return [
                "__call__",
                "__getattr__",
                "_load",
                "_module_name",
                "_attr",
                "_extras",
                "_value",
            ]
