import * as React from 'react';
import axios from 'axios';
import { Stack, StackItem, PrimaryButton, DefaultButton } from '@fluentui/react';
import { Dialog, DialogType, DialogFooter } from '@fluentui/react/lib/Dialog';
import { MANAGER_IP } from '../../static/const';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { warining, errorBadge, completed } from '../buttons/Icon';
import './customized.scss';

interface CustomizeProps {
    visible: boolean;
    copyTrialId: string;
    closeCustomizeModal: () => void;
}

interface CustomizeState {
    isShowSubmitSucceed: boolean;
    isShowSubmitFailed: boolean;
    isShowWarning: boolean;
    searchSpace: object;
    copyTrialParameter: object; // user click the trial's parameters
    customParameters: object; // customized trial, maybe user change trial's parameters
    customID: number; // submit customized trial succeed, return the new customized trial id
    changeMap: Map<string, string | number>; // store change key: value
}

class Customize extends React.Component<CustomizeProps, CustomizeState> {
    constructor(props: CustomizeProps) {
        super(props);
        this.state = {
            isShowSubmitSucceed: false,
            isShowSubmitFailed: false,
            isShowWarning: false,
            searchSpace: EXPERIMENT.searchSpace,
            copyTrialParameter: {},
            customParameters: {},
            customID: NaN,
            changeMap: new Map()
        };
    }

    getFinalVal = (event: React.ChangeEvent<HTMLInputElement>): void => {
        const { name, value } = event.target;
        const { changeMap } = this.state;
        this.setState({ changeMap: changeMap.set(name, value) });
    };

    // [submit click] user add a new trial [submit a trial]
    addNewTrial = (): void => {
        const { searchSpace, copyTrialParameter, changeMap } = this.state;
        // get user edited hyperParameter, ps: will change data type if you modify the input val
        const customized = JSON.parse(JSON.stringify(copyTrialParameter));
        // changeMap: user changed keys: values
        changeMap.forEach(function(value, key) {
            customized[key] = value;
        });

        // true: parameters are wrong
        let parametersIllegal = false;
        Object.keys(customized).map(item => {
            if (item !== 'tag') {
                // unified data type
                if (
                    (typeof copyTrialParameter[item] === 'number' && typeof customized[item] === 'string') ||
                    (typeof copyTrialParameter[item] === 'boolean' && typeof customized[item] === 'string')
                ) {
                    customized[item] = JSON.parse(customized[item]);
                }
                if (searchSpace[item] === undefined) {
                    // sometimes the schema of trial parameters is different from search space
                    // e.g. Batch Tuner
                    return;
                }
                if (searchSpace[item]._type === 'choice') {
                    if (
                        searchSpace[item]._value.find((val: string | number) => val === customized[item]) === undefined
                    ) {
                        parametersIllegal = true;
                        return;
                    }
                } else {
                    if (
                        customized[item] < searchSpace[item]._value[0] ||
                        customized[item] > searchSpace[item]._value[1]
                    ) {
                        parametersIllegal = true;
                        return;
                    }
                }
            }
        });
        if (parametersIllegal !== false) {
            // open the warning modal
            this.setState(() => ({ isShowWarning: true, customParameters: customized }));
        } else {
            // submit a customized job
            this.submitCustomize(customized);
        }
    };

    warningConfirm = (): void => {
        this.setState(() => ({ isShowWarning: false }));
        const { customParameters } = this.state;
        this.submitCustomize(customParameters);
    };

    warningCancel = (): void => {
        this.setState(() => ({ isShowWarning: false }));
    };

    submitCustomize = (customized: Record<string, any>): void => {
        // delete `tag` key
        for (const i in customized) {
            if (i === 'tag') {
                delete customized[i];
            }
        }
        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            data: customized
        })
            .then(res => {
                if (res.status === 200) {
                    this.setState(() => ({ isShowSubmitSucceed: true, customID: res.data.sequenceId }));
                    this.props.closeCustomizeModal();
                } else {
                    this.setState(() => ({ isShowSubmitFailed: true }));
                }
            })
            .catch(() => {
                this.setState(() => ({ isShowSubmitFailed: true }));
            });
    };

    closeSucceedHint = (): void => {
        // also close customized trial modal
        this.setState(() => ({ isShowSubmitSucceed: false, changeMap: new Map() }));
        this.props.closeCustomizeModal();
    };

    closeFailedHint = (): void => {
        // also close customized trial modal
        this.setState(() => ({ isShowSubmitFailed: false, changeMap: new Map() }));
        this.props.closeCustomizeModal();
    };

    componentDidMount(): void {
        const { copyTrialId } = this.props;
        if (copyTrialId !== undefined && TRIALS.getTrial(copyTrialId) !== undefined) {
            const originCopyTrialPara = TRIALS.getTrial(copyTrialId).description.parameters;
            this.setState(() => ({ copyTrialParameter: originCopyTrialPara }));
        }
    }

    componentDidUpdate(prevProps: CustomizeProps): void {
        if (this.props.copyTrialId !== prevProps.copyTrialId) {
            const { copyTrialId } = this.props;
            if (copyTrialId !== undefined && TRIALS.getTrial(copyTrialId) !== undefined) {
                const originCopyTrialPara = TRIALS.getTrial(copyTrialId).description.parameters;
                this.setState(() => ({ copyTrialParameter: originCopyTrialPara }));
            }
        }
    }

    render(): React.ReactNode {
        const { closeCustomizeModal, visible } = this.props;
        const { isShowSubmitSucceed, isShowSubmitFailed, isShowWarning, customID, copyTrialParameter } = this.state;
        const warning =
            'The parameters you set are not in our search space, this may cause the tuner to crash, Are' +
            ' you sure you want to continue submitting?';
        return (
            <Stack>
                <Dialog
                    hidden={!visible} // required field!
                    dialogContentProps={{
                        type: DialogType.largeHeader,
                        title: 'Customized trial setting',
                        subText: 'You can submit a customized trial.'
                    }}
                    modalProps={{
                        isBlocking: false,
                        styles: { main: { maxWidth: 450 } }
                    }}
                >
                    <form className='hyper-box'>
                        {Object.keys(copyTrialParameter).map(item => (
                            <Stack horizontal key={item} className='hyper-form'>
                                <StackItem styles={{ root: { minWidth: 100 } }} className='title'>
                                    {item}
                                </StackItem>
                                <StackItem className='inputs'>
                                    <input
                                        type='text'
                                        name={item}
                                        defaultValue={copyTrialParameter[item]}
                                        onChange={this.getFinalVal}
                                    />
                                </StackItem>
                            </Stack>
                        ))}
                        {/* disable [tag] because we havn't support */}
                        {/* <Stack key="tag" horizontal className="hyper-form tag-input">
                            <StackItem grow={9} className="title">Tag</StackItem>
                            <StackItem grow={15} className="inputs">
                                <input type="text" value='Customized' />
                            </StackItem>
                        </Stack> */}
                    </form>
                    <DialogFooter>
                        <PrimaryButton text='Submit' onClick={this.addNewTrial} />
                        <DefaultButton text='Cancel' onClick={closeCustomizeModal} />
                    </DialogFooter>
                </Dialog>

                {/* clone: prompt succeed or failed */}
                <Dialog
                    hidden={!isShowSubmitSucceed}
                    onDismiss={this.closeSucceedHint}
                    dialogContentProps={{
                        type: DialogType.normal,
                        title: (
                            <div className='icon-color'>
                                {completed}
                                <b>Submit successfully</b>
                            </div>
                        ),
                        closeButtonAriaLabel: 'Close',
                        subText: `You can find your customized trial by Trial No.${customID}`
                    }}
                    modalProps={{
                        isBlocking: false,
                        styles: { main: { minWidth: 500 } }
                    }}
                >
                    <DialogFooter>
                        <PrimaryButton onClick={this.closeSucceedHint} text='OK' />
                    </DialogFooter>
                </Dialog>

                <Dialog
                    hidden={!isShowSubmitFailed}
                    onDismiss={this.closeSucceedHint}
                    dialogContentProps={{
                        type: DialogType.normal,
                        title: <div className='icon-error'>{errorBadge}Submit Failed</div>,
                        closeButtonAriaLabel: 'Close',
                        subText: 'Unknown error.'
                    }}
                    modalProps={{
                        isBlocking: false,
                        styles: { main: { minWidth: 500 } }
                    }}
                >
                    <DialogFooter>
                        <PrimaryButton onClick={this.closeFailedHint} text='OK' />
                    </DialogFooter>
                </Dialog>

                {/* hyperParameter not match search space, warning modal */}
                <Dialog
                    hidden={!isShowWarning}
                    onDismiss={this.closeSucceedHint}
                    dialogContentProps={{
                        type: DialogType.normal,
                        title: <div className='icon-error'>{warining}Warning</div>,
                        closeButtonAriaLabel: 'Close',
                        subText: `${warning}`
                    }}
                    modalProps={{
                        isBlocking: false,
                        styles: { main: { minWidth: 500 } }
                    }}
                >
                    <DialogFooter>
                        <PrimaryButton onClick={this.warningConfirm} text='Confirm' />
                        <DefaultButton onClick={this.warningCancel} text='Cancel' />
                    </DialogFooter>
                </Dialog>
            </Stack>
        );
    }
}

export default Customize;
