class ConfigLoader:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self) -> dict:
        from importlib.resources import files
        import json

        config_path = files("conduit.cli").joinpath("args.json")
        json_string = config_path.read_text()
        config_dict = json.loads(json_string)
        deserialized_dict = self._deserialize(config_dict)
        return deserialized_dict

    def _deserialize(self, config_dict: dict) -> dict:
        """
        We need to rehydrate types from "str" to actual python str type, etc.
        """
        # For all positional args
        for pos_arg in config_dict["positional_args"]:
            if "type" in pos_arg.keys():
                type_name = pos_arg["type"]
                pos_arg["type"] = self._get_type_object(type_name)

        # For all commands
        for command in config_dict["commands"]:
            if "type" in command.keys():
                type_name = command["type"]
                command["type"] = self._get_type_object(type_name)
        # For all flags
        for flag in config_dict["flags"]:
            if "type" in flag.keys():
                type_name = flag["type"]
                flag["type"] = self._get_type_object(type_name)
        return config_dict

    def _get_type_object(self, type_name: str):
        match type_name:
            case "str":
                return str
            case "int":
                return int
            case "float":
                return float
            case "bool":
                return bool
            case _:
                raise ValueError(f"Unknown type: {type_name}")
