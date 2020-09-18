import * as React from 'react';
import { Dialog, DialogType, DialogFooter, Checkbox, PrimaryButton, DefaultButton } from 'office-ui-fabric-react';
import { OPERATION } from '../../static/const';

interface ChangeColumnState {
    // buffer, not saved yet
    currentSelected: string[];
}

interface ChangeColumnProps {
    hidden: boolean;
    allColumns: SimpleColumn[];  // all column List
    selectedColumns: string[];  // user selected column list
    changeColumn: (val: string[]) => void;
    onHidden: () => void;
}

interface SimpleColumn {
    key: string;  // key for management
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
    }

    onCheckboxChange = (ev: React.FormEvent<HTMLElement | HTMLInputElement> | undefined, label: string, val?: boolean): void => {
        const source: string[] = [...this.state.currentSelected];
        if (val === true) {
            if (!source.includes(label)) {
                source.push(label);
                this.setState({ currentSelected: source });
            }
        } else {
            // remove from source
            const result = source.filter((item) => item !== label);
            this.setState({ currentSelected: result });
        }
    };

    saveUserSelectColumn = (): void => {
        const { currentSelected } = this.state;
        const { allColumns } = this.props;
        const selectedColumns = allColumns.map(column => column.key).filter(key => currentSelected.includes(key));
        this.props.changeColumn(selectedColumns);
        this.hideDialog();
    }

    hideDialog = (): void => {
        this.props.onHidden();
    }

    // user exit dialog
    cancelOption = (): void => {
        // reset select column
        this.setState({ currentSelected: this.props.selectedColumns }, () => {
            this.hideDialog();
        });
    }

    render(): React.ReactNode {
        const { allColumns, hidden } = this.props;
        const { currentSelected } = this.state;
        const renderOptions: Array<CheckBoxItems> = [];
        allColumns.map(item => {
            renderOptions.push({
                label: item.name,
                checked: currentSelected.includes(item.key),
                onChange: this.makeChangeHandler(item.key)
            });
        });
        return (
            <div>
                <Dialog
                    hidden={hidden} // required field!
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
                    <div className="columns-height">
                        {renderOptions.map(item => {
                            return <Checkbox key={item.label} {...item} styles={{ root: { marginBottom: 8 } }} />
                        })}
                    </div>
                    <DialogFooter>
                        <PrimaryButton text="Save" onClick={this.saveUserSelectColumn} />
                        <DefaultButton text="Cancel" onClick={this.cancelOption} />
                    </DialogFooter>
                </Dialog>
            </div>
        );
    }
}

export default ChangeColumnComponent;
