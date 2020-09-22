import React from 'react';
import { EXPERIMENT } from '../../../static/datamodel';
import { EditExpeParamContext } from './context';
import axios from 'axios';
import { MANAGER_IP } from '../../../static/const';
import { Edit, CheckMark, Cancel } from '../../buttons/Icon';
import '../../../static/style/overview/count.scss';
import MessageInfo from '../../modals/MessageInfo';

interface EditState {
    isShowPencil: boolean;
    isShowSucceedInfo: boolean;
    defaultValue: string;
    info: string;
    typeInfo: string;
}

class EditExperimentParam extends React.Component<{}, EditState> {
    static contextType = EditExpeParamContext;
    context!: React.ContextType<typeof EditExpeParamContext>;

    private DurationInputRef = React.createRef<HTMLInputElement>();

    constructor(props) {
        super(props);
        this.state = {
            isShowPencil: true,
            isShowSucceedInfo: false,
            defaultValue: '',
            info: '',
            typeInfo: '',
        };
    }

    showPencil = () => {
        this.setState({ isShowPencil: true });
    }

    hidePencil = () => {
        this.setState({ isShowPencil: false });
        console.info('***********');
        console.info(this.DurationInputRef.current!.value);
    }

    setInputVal = (event: any) => {

        console.info('%%%%%%%');
        this.setState({ defaultValue: event.target.value });
    }

    hideSucceedInfo = (): void => {
        this.setState(() => ({ isShowSucceedInfo: false }));
    };

    /**
     * info: message content
     * typeInfo: message type: success | error...
     * continuousTime: show time, 2000ms
     */
    showMessageInfo = (info: string, typeInfo: string): void => {
        this.setState(() => ({
            info,
            typeInfo,
            isShowSucceedInfo: true
        }));
        setTimeout(this.hideSucceedInfo, 2000);
    };

    componentDidMount() {
        this.initDefaultVal();
    }

    componentDidUpdate(){
        this.initDefaultVal();
    }
    render(): React.ReactNode {
        const { isShowPencil, defaultValue, isShowSucceedInfo, typeInfo, info } = this.state;
        console.info(`333: ${defaultValue}`);
        /***
        * const CONTROLTYPE = ['MAX_EXEC_DURATION', 'MAX_TRIAL_NUM', 'TRIAL_CONCURRENCY', 'SEARCH_SPACE'];
        * [0], 'MAX_EXEC_DURATION', params.maxExecDuration
        * [1], 'MAX_TRIAL_NUM', params.maxTrialNum
        * [2], 'TRIAL_CONCURRENCY', params.trialConcurrency 
        */

        return (
            <EditExpeParamContext.Consumer>
                {(value): React.ReactNode => (
                    <React.Fragment>
                        <p>{value.title}</p>
                        <div>
                            <input
                                // type='number'
                                className='durationInput'
                                // defaultValue={defaultValue}
                                ref={this.DurationInputRef}
                                disabled={isShowPencil ? true : false}
                                value={defaultValue}
                                onChange={this.setInputVal}
                            />{value.unit}
                            {isShowPencil &&
                                <span className='edit' onClick={this.hidePencil
                                }>{Edit}</span>}

                            {!isShowPencil &&
                                <span className='series'>
                                    <span className='confirm' onClick={this.confirmEdit}>{CheckMark}</span>
                                    <span className='cancel' onClick={this.cancleEdit}>{Cancel}</span>
                                </span>}

                            {isShowSucceedInfo && <MessageInfo className='info' typeInfo={typeInfo} info={info} />}
                        </div>
                    </React.Fragment>
                )
                }
            </EditExpeParamContext.Consumer>

        );
    }

    initDefaultVal = () => {
        const { title } = this.context;
        const maxExecDuration = EXPERIMENT.profile.params.maxExecDuration;
        if (title === 'Max duration') {
            this.DurationInputRef.current!.value = maxExecDuration.toString();
        } else if (title === 'Max trial numbers') {
            this.DurationInputRef.current!.value = EXPERIMENT.profile.params.maxTrialNum.toString();
        } else {
            this.DurationInputRef.current!.value = EXPERIMENT.profile.params.trialConcurrency.toString();
        }
    }

    cancleEdit = (): void => {
        this.showPencil();
    }

    confirmEdit = async (): Promise<void> => {
        const {title} = this.context;
        const maxExecDuration = EXPERIMENT.profile.params.maxExecDuration;
        const { field, editType } = this.context;
        const userInput: string = this.DurationInputRef.current!.value;
        console.info(`userInput default: ${userInput}`);
        if (!userInput.match(/^[1-9]\d*$/)) {
            this.showMessageInfo('Please enter a positive integer!', 'error');
            return;
        }
        let val = '';
        if (title === 'Max duration') {
            val = maxExecDuration.toString();
        } else if (title === 'Max trial numbers') {
            val = EXPERIMENT.profile.params.maxTrialNum.toString();
        } else {
            val = EXPERIMENT.profile.params.trialConcurrency.toString();
        }
        if (userInput === val) {
            this.showMessageInfo(`Trial ${field} has not changed`, 'error');
            return;
        }
        const newValue = parseInt(userInput, 10);
        const newProfile = Object.assign({}, EXPERIMENT.profile);
        newProfile.params[field] = newValue;

        // rest api, modify trial concurrency value
        try {
            const res = await axios.put(`${MANAGER_IP}/experiment`, newProfile, {
                // eslint-disable-next-line @typescript-eslint/camelcase
                params: { update_type: editType }
            });
            if (res.status === 200) {
                this.showMessageInfo(`Successfully updated ${field}`, 'success');
            }
        } catch (error) {
            if (error.response && error.response.data.error) {
                this.showMessageInfo(`Failed to update trial ${field}\n${error.response.data.error}`, 'error');
            }
            else if (error.response) {
                this.showMessageInfo(
                    `Failed to update trial ${field}\nServer responsed ${error.response.status}`, 'error'
                );
            } else if (error.message) {
                this.showMessageInfo(`Failed to update trial ${field}\n${error.message}`, 'error');
            } else {
                this.showMessageInfo(`Failed to update trial ${field}\nUnknown error`, 'error');
            }
        }
        this.showPencil();
    }
}

export default EditExperimentParam;
