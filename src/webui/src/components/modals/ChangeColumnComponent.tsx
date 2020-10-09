import * as React from 'react';
import { Dialog, DialogType, DialogFooter, Checkbox, PrimaryButton, DefaultButton } from '@fluentui/react';

interface ChangeColumnState {
    // buffer, not saved yet
    currentSelected: string[];
}

interface ChangeColumnProps {
    allColumns: SimpleColumn[]; // all column List
    selectedColumns: string[]; // user selected column list
    onSelectedChange: (val: string[]) => void;
    onHideDialog: () => void;
    minSelected?: number;
}

interface SimpleColumn {
    key: string; // key for management
    name: string; // name to display
}

interface CheckBoxItems {
    label: string;
    checked: boolean;
    onChange: () => void;
}

class ChangeColumnComponent extends React.Component<ChangeColumnProps, ChangeColumnState> {
    constructor(props: ChangeColumnProps) {
        super(props);
        this.state = {
            currentSelected: this.props.selectedColumns
        };
    }

    makeChangeHandler = (label: string): any => {
        return (ev: any, checked: boolean): void => this.onCheckboxChange(ev, label, checked);
    };

    onCheckboxChange = (
        ev: React.FormEvent<HTMLElement | HTMLInputElement> | undefined,
        label: string,
        val?: boolean
    ): void => {
        const source: string[] = [...this.state.currentSelected];
        if (val === true) {
            if (!source.includes(label)) {
                source.push(label);
                this.setState({ currentSelected: source });
            }
        } else {
            // remove from source
            const result = source.filter(item => item !== label);
            this.setState({ currentSelected: result });
        }
    };

    saveUserSelectColumn = (): void => {
        const { currentSelected } = this.state;
        const { allColumns, onSelectedChange } = this.props;
        const selectedColumns = allColumns.map(column => column.key).filter(key => currentSelected.includes(key));
        onSelectedChange(selectedColumns);
        this.hideDialog();
    };

    // user exit dialog
    cancelOption = (): void => {
        // reset select column
        this.setState({ currentSelected: this.props.selectedColumns }, () => {
            this.hideDialog();
        });
    };

    private hideDialog = (): void => {
        this.props.onHideDialog();
    };

    render(): React.ReactNode {
        const { allColumns, minSelected } = this.props;
        const { currentSelected } = this.state;
        return (
            <div>
                <Dialog
                    hidden={false}
                    dialogContentProps={{
                        type: DialogType.largeHeader,
                        title: 'Customize columns',
                        subText: 'You can choose which columns you wish to see.'
                    }}
                    modalProps={{
                        isBlocking: false,
                        styles: { main: { maxWidth: 450 } }
                    }}
                >
                    <div className='columns-height'>
                        {allColumns.map(item => (
                            <Checkbox
                                key={item.key}
                                label={item.name}
                                checked={currentSelected.includes(item.key)}
                                onChange={this.makeChangeHandler(item.key)}
                                styles={{ root: { marginBottom: 8 } }}
                            />
                        ))}
                    </div>
                    <DialogFooter>
                        <PrimaryButton
                            text='Save'
                            onClick={this.saveUserSelectColumn}
                            disabled={currentSelected.length < (minSelected === undefined ? 1 : minSelected)}
                        />
                        <DefaultButton text='Cancel' onClick={this.cancelOption} />
                    </DialogFooter>
                </Dialog>
            </div>
        );
    }
}

export default ChangeColumnComponent;
