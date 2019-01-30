import * as React from 'react';
import axios from 'axios';
import { message } from 'antd';
import { MANAGER_IP } from '../const';
import '../style/tensor.css';

interface TensorState {
    urlTensor: string;
    idTensor: string;
}

message.config({
    top: 250,
    duration: 2,
});

class Tensor extends React.Component<{}, TensorState> {

    public _isMounted = false;

    constructor(props: {}) {
        super(props);
        this.state = {
            urlTensor: '',
            idTensor: ''
        };
    }

    geturl(): void {
        Object.keys(this.props).forEach(item => {
            if (item === 'location') {
                let tensorId = this.props[item].state;
                if (tensorId !== undefined && this._isMounted) {
                    this.setState({ idTensor: tensorId }, () => {
                        axios(`${MANAGER_IP}/tensorboard`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json;charset=utf-8'
                            },
                            params: {
                                job_ids: tensorId
                            }
                        }).then(res => {
                            if (res.status === 200) {
                                setTimeout(
                                    () => {
                                        const url = new URL(res.data.endPoint);
                                        if (url.hostname === 'localhost') {
                                            url.hostname = window.location.hostname;
                                        }
                                        this.setState(
                                            { urlTensor: url.href }, 
                                            () => message.success('Successful send'));
                                    },
                                    1000);
                            } else {
                                message.error('fail to link to tensorboard');
                            }
                        });
                    });
                } else {
                    message.warning('Please link to Trial Status page to select a trial!');
                }
            }
        });
    }

    componentDidMount() {
        this._isMounted = true;
        this.geturl();
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { urlTensor } = this.state;
        return (
            <div className="tensor">
                <div className="title">TensorBoard</div>
                <div className="tenhttpage">
                    <iframe
                        frameBorder="no"
                        src={urlTensor}
                        sandbox="allow-scripts allow-same-origin"
                    />
                </div>
            </div>
        );
    }
}

export default Tensor;