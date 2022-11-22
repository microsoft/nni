import React, { useState } from 'react';
import { Dialog, DialogType, DialogFooter, Checkbox, PrimaryButton, DefaultButton } from '@fluentui/react';
import { EXPERIMENT } from '@static/datamodel';
import { Storage } from '@model/localStorage';

/**
 * changeColumnComponent file is for [customized table column, customized hyper-parameter graph yAxis]
 * and currently it uses localstorage to store the customized results
 */

interface ChangeColumnProps {
    allColumns: SimpleColumn[]; // all column List
    selectedColumns: string[]; // user selected column list
    onSelectedChange: (val: string[]) => void;
    onHideDialog: () => void;
    minSelected?: number;
    whichComponent: string; // which component use this component
}

interface SimpleColumn {
    key: string; // key for management
    name: string; // name to display
}

//interface CheckBoxItems {
//    label: string;
//    checked: boolean;
//    onChange: () => void;
//}
const ChangeColumnComponent = (props: ChangeColumnProps): any => {
    const { selectedColumns, allColumns, minSelected, onHideDialog } = props;
    const [currentSelected, setCurrentSelected] = useState(selectedColumns as string[]);
    const onCheckboxChange = (
        ev: React.FormEvent<HTMLElement | HTMLInputElement> | undefined,
        label: string,
        val?: boolean
    ): void => {
        const source: string[] = [...currentSelected];
        if (val === true) {
            if (!source.includes(label)) {
                source.push(label);
                setCurrentSelected(source);
            }
        } else {
            // remove from source
            const result = source.filter(item => item !== label);
            setCurrentSelected(result);
        }
    };
    const makeChangeHandler = (label: string): any => {
        return (ev: any, checked: boolean): void => onCheckboxChange(ev, label, checked);
    };

    const saveUserSelectColumn = (): void => {
        const { allColumns, onSelectedChange, whichComponent } = props;
        const selectedColumns = allColumns.map(column => column.key).filter(key => currentSelected.includes(key));
        onSelectedChange(selectedColumns);
        if (whichComponent === 'table') {
            const storage = new Storage(
                `${EXPERIMENT.profile.id}_columns`,
                JSON.stringify(selectedColumns),
                30 * 24 * 60 * 60 * 1000
            );
            storage.setValue();
        } else {
            const storage = new Storage(
                `${EXPERIMENT.profile.id}_paraColumns`,
                JSON.stringify(selectedColumns),
                30 * 24 * 60 * 60 * 1000
            );
            storage.setValue();
        }
        onHideDialog();
    };

    // user exit dialog
    const cancelOption = (): void => {
        // reset select column
        setCurrentSelected(selectedColumns); // todo: useeffect 里写回调; 或者延时函数里关闭窗口
        onHideDialog(); // todo 测试这里写入数据是否正常
        // this.setState({ currentSelected: this.props.selectedColumns }, () => {
        //     this.hideDialog();
        // });
    };

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
                            onChange={makeChangeHandler(item.key)}
                            styles={{ root: { marginBottom: 8 } }}
                        />
                    ))}
                </div>
                <DialogFooter>
                    <PrimaryButton
                        text='Save'
                        onClick={saveUserSelectColumn}
                        disabled={currentSelected.length < (minSelected ?? 1)}
                    />
                    <DefaultButton text='Cancel' onClick={cancelOption} />
                </DialogFooter>
            </Dialog>
        </div>
    );
};

export default ChangeColumnComponent;
