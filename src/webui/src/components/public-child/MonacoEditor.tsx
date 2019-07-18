import * as React from 'react';
import { Spin } from 'antd';
import { DRAWEROPTION } from '../../static/const';
import MonacoEditor from 'react-monaco-editor';

interface MonacoEditorProps {
    content: string;
    loading: boolean;
}

class MonacoHTML extends React.Component<MonacoEditorProps, {}> {

    public _isMonacoMount: boolean;

    constructor(props: MonacoEditorProps) {
        super(props);
    }

    componentDidMount() {
        this._isMonacoMount = true;
    }

    componentWillUnmount() {
        this._isMonacoMount = false;
    }

    render() {
        const { content, loading } = this.props;
        const heights: number = window.innerHeight - 48;
        return (
            <div className="just-for-log">
                <Spin
                    // tip="Loading..."
                    style={{ width: '100%', height: heights * 0.9 }}
                    spinning={loading}
                >
                    <MonacoEditor
                        width="100%"
                        height={heights * 0.9}
                        language="json"
                        value={content}
                        options={DRAWEROPTION}
                    />
                </Spin>
            </div>
        );
    }
}

export default MonacoHTML;
