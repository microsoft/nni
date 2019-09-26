import * as React from 'react';
import { Button, Row } from 'antd';

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

    save = () => {
        if (this.input.current !== null) {
            this.props.updateValue(this.input.current.value);
            this.setState({ editting: false });
        }
    }

    cancel = () => {
        this.setState({ editting: false });
    }

    edit = () => {
        this.setState({ editting: true });
    }

    render() {
        if (this.state.editting) {
            return (
                <Row className="inputBox">
                    <input
                        type="number"
                        className="concurrencyInput"
                        defaultValue={this.props.value.toString()}
                        ref={this.input}
                    />
                    <Button
                        type="primary"
                        className="tableButton editStyle"
                        onClick={this.save}
                    >
                        Save
                    </Button>
                    <Button
                        type="primary"
                        onClick={this.cancel}
                        style={{ display: 'inline-block', marginLeft: 1 }}
                        className="tableButton editStyle"
                    >
                        Cancel
                    </Button>
                </Row>
            );
        } else {
            return (
                <Row className="inputBox">
                    <input
                        type="number"
                        className="concurrencyInput"
                        disabled={true}
                        value={this.props.value}
                    />
                    <Button
                        type="primary"
                        className="tableButton editStyle"
                        onClick={this.edit}
                    >
                        Edit
                    </Button>
                </Row>
            );
        }
    }
}

export default ConcurrencyInput;
