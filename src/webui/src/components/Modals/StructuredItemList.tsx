import * as React from 'react';
import { Dialog, DialogType, DialogFooter, Checkbox, PrimaryButton, DefaultButton } from 'office-ui-fabric-react';
import { OPERATION } from '../../static/const';

interface ChangeColumnState {
    userSelectColumnList: string[];
    originSelectColumnList: string[];
}

interface ChangeColumnProps {
    isHideDialog: boolean;
    showColumn: string[]; // all column List
    selectedColumn: string[]; // user selected column list
    changeColumn: (val: string[]) => void;
    hideShowColumnDialog: () => void;
}

interface CheckBoxItems {
    label: string;
    checked: boolean;
    onChange: () => void;
}

class StructuredItemList extends React.Component<ChangeColumnProps, ChangeColumnState> {

    constructor(props: ChangeColumnProps) {
        super(props);
        this.state = { userSelectColumnList: this.props.selectedColumn, originSelectColumnList: this.props.selectedColumn };
    }

    makeChangeHandler = (label: string): any => {
        return (ev: any, checked: boolean): void => this.onCheckboxChange(ev, label, checked);
    }

    onCheckboxChange = (ev: React.FormEvent<HTMLElement | HTMLInputElement> | undefined, label: string, val?: boolean, ): void => {
        const source: string[] = JSON.parse(JSON.stringify(this.state.userSelectColumnList));
        if (val === true) {
            if (!source.includes(label)) {
                source.push(label);
                this.setState(() => ({ userSelectColumnList: source }));
            }
        } else {
            if (source.includes(label)) {
                // remove from source
                const result = source.filter((item) => item !== label);
                this.setState(() => ({ userSelectColumnList: result }));
            }
        }
    };

    saveUserSelectColumn = (): void => {
        const { userSelectColumnList } = this.state;
        const { showColumn } = this.props;
        // sort by Trial No. | ID | Duration | Start Time | End Time | ...
        const sortColumn: string[] = [];
        /**
         * 
         * TODO: use this function to refactor sort column
         * search space might orderless
            showColumn.map(item => {
                userSelectColumnList.map(key => {
                    if (item === key || key.includes('search space')) {
                        if (!sortColumn.includes(key)) {
                            sortColumn.push(key);
                        }
                    }
                });
            });
         */
        // push ![Operation] ![search space] column
        showColumn.map(item => {
            userSelectColumnList.map(key => {
                if (item === key && item !== OPERATION) {
                    sortColumn.push(key);
                }
            });
        });
        // push search space key
        userSelectColumnList.map(index => {
            if (index.includes('search space')) {
                if (!sortColumn.includes(index)) {
                    sortColumn.push(index);
                }
            }
        });
        // push Operation
        if (userSelectColumnList.includes(OPERATION)) {
            sortColumn.push(OPERATION);
        }
        this.props.changeColumn(sortColumn);
        this.hideDialog(); // hide dialog
    }

    hideDialog = (): void => {
        this.props.hideShowColumnDialog();
    }

    // user exit dialog
    cancelOption = (): void => {
        // reset select column
        const { originSelectColumnList } = this.state;
        this.setState({ userSelectColumnList: originSelectColumnList }, () => {
            this.hideDialog();
        });
    }

    render(): React.ReactNode {
        const { showColumn, isHideDialog } = this.props;
        const { userSelectColumnList } = this.state;
        const renderOptions: Array<CheckBoxItems> = [];
        showColumn.map(item => {
            if (userSelectColumnList.includes(item)) {
                // selected column name
                renderOptions.push({ label: item, checked: true, onChange: this.makeChangeHandler(item) });
            } else {
                renderOptions.push({ label: item, checked: false, onChange: this.makeChangeHandler(item) });
            }
        });
        return (
            <div>
                <Dialog
                    hidden={isHideDialog} // required field!
                    dialogContentProps={{
                        type: DialogType.largeHeader,
                        title: 'Change columns'
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

export default StructuredItemList;
