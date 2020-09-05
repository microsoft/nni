import * as React from 'react';
import { Stack, PrimaryButton } from 'office-ui-fabric-react';

interface ConcurrencyInputProps {
    value: number;
    updateValue: (val: string) => void;
}

interface ConcurrencyInputStates {
    editting: boolean;
}

class ConcurrencyInput extends React.Component<ConcurrencyInputProps, ConcurrencyInputStates> {
    private input = React.createRef<HTMLInputElement>();

    constructor(props: ConcurrencyInputProps) {
        super(props);
        this.state = { editting: false };
    }

    save = (): void => {
        if (this.input.current !== null) {
            this.props.updateValue(this.input.current.value);
            this.setState({ editting: false });
        }
    }

    cancel = (): void => {
        this.setState({ editting: false });
    }

    edit = (): void => {
        this.setState({ editting: true });
    }

    render(): React.ReactNode {
        if (this.state.editting) {
            return (
                <Stack horizontal className="inputBox">
                    <input
                        type="number"
                        className="concurrencyInput"
                        defaultValue={this.props.value.toString()}
                        ref={this.input}
                    />
                    <PrimaryButton
                        text="Save"
                        onClick={this.save}
                    />
                    <PrimaryButton
                        text="Cancel"
                        style={{ display: 'inline-block', marginLeft: 1 }}
                        onClick={this.cancel}
                    />
                </Stack>
            );
        } else {
            return (
                <Stack horizontal className="inputBox">
                    <input
                        type="number"
                        className="concurrencyInput"
                        disabled={true}
                        value={this.props.value}
                    />
                    <PrimaryButton
                        text="Edit"
                        onClick={this.edit}
                    />
                </Stack>
            );
        }
    }
}

export default ConcurrencyInput;
